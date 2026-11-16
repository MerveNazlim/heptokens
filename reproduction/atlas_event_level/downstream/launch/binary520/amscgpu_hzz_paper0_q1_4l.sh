#!/usr/bin/env bash
# Three independent binary fits on the same sealed ATLAS four-lepton cohort.
set -euo pipefail
CLUSTER_ID=${1:?Missing cluster ID}
PROC_ID=${2:?Missing process ID}
ARM=${3:?Missing arm}
[[ "$CLUSTER_ID" =~ ^[0-9]+$ && "$PROC_ID" =~ ^[0-9]+$ && "$ARM" =~ ^(scratch|finetune|continuous_finetune)$ ]]
for digest in "${EXPECTED_RUNTIME_SHA256:?}" "${EXPECTED_DATA_SHA256:?}" "${EXPECTED_MANIFEST_SHA256:?}" "${EXPECTED_MEMBERSHIP_SHA256:?}"; do
  [[ "$digest" =~ ^[a-f0-9]{64}$ ]]
done

SCRATCH=${_CONDOR_SCRATCH_DIR:?Missing Condor scratch}
WORK="$SCRATCH/hzz_paper0_q1_4l"
RESULTS="$WORK/results"
REPO="$WORK/project"
DATASET="$WORK/prepared"
if [[ "$ARM" == continuous_finetune ]]; then ARCHIVE=continuous.tar.gz; else ARCHIVE=q1.tar.gz; fi
FIT="$WORK/fit"
PREFIX=gs://fcc-teststorage/bucket_results/foundation_pretrain
DATA="$PREFIX/downstream_data/hzz_paper0_q1_4l_v1"
DEST="$PREFIX/downstream/hzz_paper0_q1_4l_v1/$ARM/cluster_${CLUSTER_ID}_proc_${PROC_ID}"
KEY="$SCRATCH/fccstorageuser.json"
export CLOUDSDK_CONFIG="$WORK/gcloud-config" PIXI_HOME="$WORK/pixi"
export XDG_CACHE_HOME="$WORK/cache" XDG_CONFIG_HOME="$WORK/config" UV_CACHE_DIR="$WORK/cache/uv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=offline
mkdir -p "$RESULTS" "$REPO" "$CLOUDSDK_CONFIG"
exec > >(tee -a "$RESULTS/setup.log") 2>&1
AUTH_READY=0
STAGE=setup
STARTED=$(date --iso-8601=seconds)

finish() {
  local rc=$? sync_rc status_rc
  trap - EXIT
  set +e
  if [[ "$AUTH_READY" == 1 ]]; then
    gcloud storage rsync --recursive "$RESULTS" "$DEST"
    sync_rc=$?
    if [[ "$rc" == 0 && "$sync_rc" != 0 ]]; then rc=$sync_rc; STAGE=upload_results; fi
    {
      printf 'exit_code=%s\nstage=%s\narm=%s\n' "$rc" "$STAGE" "$ARM"
      printf 'started=%s\nfinished=%s\nworker=%s\n' "$STARTED" "$(date --iso-8601=seconds)" "$(hostname)"
      printf 'runtime_sha256=%s\ndata_archive=%s\ndata_sha256=%s\nmanifest_sha256=%s\nmembership_sha256=%s\n' \
        "$EXPECTED_RUNTIME_SHA256" "$ARCHIVE" "$EXPECTED_DATA_SHA256" "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_MEMBERSHIP_SHA256"
    } > "$WORK/job_status.txt"
    gcloud storage cp --if-generation-match=0 "$WORK/job_status.txt" "$DEST/job_status.txt"
    status_rc=$?
    if [[ "$rc" == 0 && "$status_rc" != 0 ]]; then rc=$status_rc; fi
  fi
  if [[ "$CLOUDSDK_CONFIG" == "$WORK/gcloud-config" && -d "$CLOUDSDK_CONFIG" ]]; then
    rm -rf -- "$CLOUDSDK_CONFIG"
  fi
  exit "$rc"
}
trap finish EXIT

check_sha() {
  [[ "$2" =~ ^[a-f0-9]{64}$ && "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || {
    printf 'Checksum mismatch: %s\n' "$1" >&2; return 2;
  }
}
fetch() {
  gcloud storage cp "$1" "$2"
  check_sha "$2" "$3"
}
extract() {
  python3 - "$1" "$2" "${3:-}" <<'PY'
import pathlib, sys, tarfile

with tarfile.open(sys.argv[1], 'r:gz') as archive:
    members = archive.getmembers()
    seen = set()
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        if (not name.parts or name.is_absolute() or '..' in name.parts or name in seen
                or not (member.isfile() or member.isdir()) or name.name == 'fccstorageuser.json'
                or (sys.argv[3] and name.parts[0] != sys.argv[3])):
            raise ValueError('Unsafe archive member: ' + member.name)
        seen.add(name)
    archive.extractall(sys.argv[2], members=members)
PY
}

command -v gcloud >/dev/null || { echo 'gcloud is required on the worker'; exit 2; }
test -s "$KEY"
chmod 600 "$KEY"
gcloud auth activate-service-account --key-file="$KEY"
AUTH_READY=1
STAGE=verify_inputs
if gcloud storage ls "$DEST/job_status.txt" >/dev/null 2>&1; then
  AUTH_READY=0
  echo 'Refusing to overwrite previously published results'; exit 2
fi
fetch "$DATA/runtime.tar.gz" "$WORK/runtime.tar.gz" "$EXPECTED_RUNTIME_SHA256"
extract "$WORK/runtime.tar.gz" "$REPO"
fetch "$DATA/$ARCHIVE" "$WORK/$ARCHIVE" "$EXPECTED_DATA_SHA256"
extract "$WORK/$ARCHIVE" "$WORK" prepared
check_sha "$DATASET/manifest.json" "$EXPECTED_MANIFEST_SHA256"
check_sha "$DATASET/membership.npz" "$EXPECTED_MEMBERSHIP_SHA256"
cp "$DATASET/manifest.json" "$RESULTS/sealed_input_manifest.json"
cd "$REPO"
cp condor/pixi-matched-multiclass.toml pixi.toml
cp condor/pixi-matched-multiclass.lock pixi.lock
export PYTHONPATH="$REPO/src:$REPO/scripts"

STAGE=setup_environment
if command -v pixi >/dev/null 2>&1; then
  PIXI=$(command -v pixi)
else
  curl --retry 5 --retry-all-errors --retry-delay 5 -fsSL https://pixi.sh/install.sh -o "$WORK/install-pixi.sh"
  for attempt in 1 2 3; do
    if PIXI_NO_PATH_UPDATE=1 bash "$WORK/install-pixi.sh" && [[ -x "$PIXI_HOME/bin/pixi" ]]; then break; fi
    [[ "$attempt" != 3 ]] || { echo 'Pixi bootstrap failed'; exit 2; }
    sleep 10
  done
  PIXI="$PIXI_HOME/bin/pixi"
fi
"$PIXI" install --manifest-path pixi.toml --platform linux-64 --locked 2>&1 | tee "$RESULTS/pixi_install.log"
RUN=(env "LD_LIBRARY_PATH=$REPO/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PIXI" run --manifest-path pixi.toml --platform linux-64 --locked)
"${RUN[@]}" python - <<'PY' | tee "$RESULTS/runtime.txt"
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.__version__.startswith('2.11.0') and torch.version.cuda == '13.0'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

SOURCE_ARGS=()
if [[ "$ARM" != scratch ]]; then
  STAGE=verify_pretrained_source
  PLAN="$REPO/multiclass_controls_plan.json"
  python3 - "$PLAN" "$ARM" > "$WORK/source_files.tsv" <<'PY'
import json, re, sys

family, job, config_id = {
    'finetune': ('q1_projected', '130.0', 'q1_projected_v1_linear_1x32'),
    'continuous_finetune': ('continuous_flat', '133.0', 'continuous_flat_v1_schema'),
}[sys.argv[2]]
source = json.load(open(sys.argv[1]))['sources'][family + '_p42']
if ((source['model_family'], source['pretrain_seed'], source['pretraining_job']) != (family, 42, job)
        or source['pretraining_configuration_id'] != config_id):
    raise ValueError('Not the pinned seed-42 backbone: ' + family)
for key, name in (('checkpoint', 'checkpoint.ckpt'), ('pretrain_config', 'config.yaml'),
                  ('pretrain_status', 'status.txt'), ('pretraining_vocabulary_shard', 'vocabulary.parquet')):
    if (key == 'pretraining_vocabulary_shard' and family == 'continuous_flat'
            and key not in source and key + '_sha256' not in source):
        continue
    uri, digest = source[key], source[key + '_sha256']
    if not (uri.startswith('gs://fcc-teststorage/') and not any(c.isspace() for c in uri)
            and re.fullmatch('[0-9a-f]{64}', digest)):
        raise ValueError('Unpinned pretraining input: ' + key)
    print(uri, name, digest, sep='\t')
PY
  SOURCE="$WORK/pretrained"
  mkdir "$SOURCE"
  while IFS=$'\t' read -r uri name digest; do
    fetch "$uri" "$SOURCE/$name" "$digest"
  done < "$WORK/source_files.tsv"
  SOURCE_ARGS=(--source-dir "$SOURCE" --plan "$PLAN")
fi

STAGE=fit_and_evaluate
"${RUN[@]}" python scripts/train_atlas_paper0_q1_4l.py \
  --dataset-dir "$DATASET" --output-dir "$FIT" --arm "$ARM" \
  --manifest-sha256 "$EXPECTED_MANIFEST_SHA256" "${SOURCE_ARGS[@]}" \
  2>&1 | tee "$RESULTS/training.log"
test -s "$FIT/completion.json"
shopt -s dotglob nullglob
for artifact in "$FIT"/*; do
  target="$RESULTS/${artifact##*/}"
  [[ ! -e "$target" && ! -L "$target" ]] || { echo "Training output collision: $target"; exit 2; }
  mv -- "$artifact" "$target"
done
STAGE=complete
