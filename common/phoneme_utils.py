from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from phonemizer import phonemize
from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIF = PROJECT_ROOT / "containers" / "phonemizer_lt.sif"


def _ensure_list(texts):
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _host_has_espeak() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _phonemize_host(texts, language: str = "lt") -> list[str]:
    texts = _ensure_list(texts)
    out = phonemize(
        texts,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        njobs=1,
    )
    if isinstance(out, str):
        return [out]
    return list(out)


def _phonemize_container(texts, language: str = "lt") -> list[str]:
    texts = _ensure_list(texts)
    sif_path = Path(os.environ.get("PHONEMIZER_LT_SIF", str(DEFAULT_SIF))).expanduser().resolve()

    if not sif_path.exists():
        raise RuntimeError(
            f"Nerastas phonemizer konteineris: {sif_path}. "
            "Nurodyk PHONEMIZER_LT_SIF arba įdėk konteinerį į bakis/containers/phonemizer_lt.sif"
        )

    workdir = Path(os.environ.get("PHONEMIZER_WORKDIR", str(Path.home() / "workdir"))).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps({"texts": texts, "language": language}, ensure_ascii=False)

    inner_code = r'''
import json
import sys
from phonemizer import phonemize

payload = json.loads(sys.stdin.read())
out = phonemize(
    payload["texts"],
    language=payload["language"],
    backend="espeak",
    strip=True,
    preserve_punctuation=False,
    with_stress=False,
    njobs=1,
)
if isinstance(out, str):
    out = [out]
print(json.dumps(out, ensure_ascii=False))
'''.strip()

    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", "/tmp")

    cmd = [
        "singularity",
        "exec",
        "-H",
        f"{workdir}:{Path.home()}",
        str(sif_path),
        "python3",
        "-c",
        inner_code,
    ]

    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Nepavyko paleisti phonemizer konteineryje.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("Phonemizer konteineris negrąžino išvesties.")

    last_line = stdout.splitlines()[-1]
    return json.loads(last_line)


def phonemize_texts(texts, language: str = "lt") -> list[str]:
    if _host_has_espeak():
        return _phonemize_host(texts, language=language)
    return _phonemize_container(texts, language=language)


def normalized_ped(reference: str, prediction: str, language: str = "lt") -> float:
    ref_ph, pred_ph = phonemize_texts([reference, prediction], language=language)
    return Levenshtein.distance(ref_ph, pred_ph) / max(1, len(ref_ph))