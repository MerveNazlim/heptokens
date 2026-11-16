#!/usr/bin/env bash
# Frozen downstream protocol; only the preselected Paper0-core cohort differs.
set -euo pipefail
CLUSTER_ID=${1:?Missing ClusterId}
PROC_ID=${2:?Missing ProcId}
MODE=${3:?Missing mode}
RUN_KEY=${4:?Missing run key}
DATA_KIND=${5:?Missing dataset kind}
[[ "$CLUSTER_ID" =~ ^[0-9]+$ && "$PROC_ID" =~ ^[0-9]+$ && "$RUN_KEY" =~ ^[a-z0-9_]+$ ]]
[[ ( "$MODE" == train && "$DATA_KIND" =~ ^(q1|q4|q8|continuous)$ ) || \
   ( "$MODE" == plot && "$RUN_KEY" == analysis && "$DATA_KIND" == common ) ]]
for digest in "${EXPECTED_RUNTIME_SHA256:?}" "${EXPECTED_COMMON_SHA256:?}" "${EXPECTED_MEMBERSHIP_SHA256:?}" \
  "${EXPECTED_SOURCE_MANIFEST_SHA256:?}"; do
  [[ "$digest" =~ ^[a-f0-9]{64}$ ]]
done
if [[ "$MODE" == train ]]; then [[ "${EXPECTED_DATA_SHA256:?}" =~ ^[a-f0-9]{64}$ ]]; fi
SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:?Missing Condor worker scratch directory}
WORK_DIR="$SCRATCH_DIR/hzz_4l_core_multiclass_work"
REPO="$WORK_DIR/project"
RESULTS="$WORK_DIR/results"
DATASET="$WORK_DIR/dataset"
PREFIX=gs://fcc-teststorage/bucket_results/foundation_pretrain
DATA="$PREFIX/downstream_data/hzz_4l_core_multiclass_v1"
RESULT_PREFIX="$PREFIX/downstream/hzz_4l_core_multiclass_v1"
DEST="$RESULT_PREFIX/$RUN_KEY/cluster_${CLUSTER_ID}_proc_${PROC_ID}"
export CLOUDSDK_CONFIG="$WORK_DIR/gcloud-config" PIXI_HOME="$WORK_DIR/pixi"
export XDG_CACHE_HOME="$WORK_DIR/cache" XDG_CONFIG_HOME="$WORK_DIR/config"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=offline
mkdir -p "$RESULTS" "$CLOUDSDK_CONFIG" "$REPO" "$DATASET"
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
    if [[ "$rc" == 0 && "$sync_rc" != 0 ]]; then rc=$sync_rc; STAGE=final_result_upload_failed; fi
    {
      printf 'exit_code=%s\nstage=%s\nmode=%s\nrun_key=%s\ndata_kind=%s\n' "$rc" "$STAGE" "$MODE" "$RUN_KEY" "$DATA_KIND"
      printf 'started=%s\nfinished=%s\nworker=%s\n' "$STARTED" "$(date --iso-8601=seconds)" "$(hostname)"
      printf 'runtime_sha256=%s\ncommon_sha256=%s\nmembership_sha256=%s\n' \
        "$EXPECTED_RUNTIME_SHA256" "$EXPECTED_COMMON_SHA256" "$EXPECTED_MEMBERSHIP_SHA256"
    } > "$WORK_DIR/job_status.txt"
    gcloud storage cp --if-generation-match=0 "$WORK_DIR/job_status.txt" "$DEST/job_status.txt"
    status_rc=$?
    if [[ "$rc" == 0 && "$status_rc" != 0 ]]; then rc=$status_rc; fi
  fi
  if [[ "$CLOUDSDK_CONFIG" == "$WORK_DIR/gcloud-config" && -d "$CLOUDSDK_CONFIG" ]]; then
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
    allowed = set(sys.argv[3].split(',')) if sys.argv[3] else None
    for m in members:
        p = pathlib.PurePosixPath(m.name)
        if (not p.parts or p.is_absolute() or '..' in p.parts or p in seen
                or not (m.isfile() or m.isdir()) or p.name == 'fccstorageuser.json'
                or (allowed is not None and p.parts[0] not in allowed)):
            raise ValueError('Unsafe/unexpected archive member: ' + m.name)
        seen.add(p)
    archive.extractall(sys.argv[2], members=members)
PY
}

command -v gcloud >/dev/null || { echo 'gcloud is required on the worker'; exit 2; }
KEY="$SCRATCH_DIR/fccstorageuser.json"
test -s "$KEY"
chmod 600 "$KEY"
gcloud auth activate-service-account --key-file="$KEY"
AUTH_READY=1
STAGE=verify_immutable_inputs
if gcloud storage ls "$DEST/job_status.txt" >/dev/null 2>&1; then
  AUTH_READY=0
  echo 'Refusing an already published job result'; exit 2
fi
fetch "$DATA/runtime.tar.gz" "$WORK_DIR/runtime.tar.gz" "$EXPECTED_RUNTIME_SHA256"
extract "$WORK_DIR/runtime.tar.gz" "$REPO"
fetch "$DATA/common.tar.gz" "$WORK_DIR/common.tar.gz" "$EXPECTED_COMMON_SHA256"
extract "$WORK_DIR/common.tar.gz" "$DATASET" selection,canonical_index
check_sha "$DATASET/selection/membership.npz" "$EXPECTED_MEMBERSHIP_SHA256"
check_sha "$DATASET/selection/source_manifest.json" "$EXPECTED_SOURCE_MANIFEST_SHA256"
cp "$REPO/hzz_4l_multiclass_plan.json" "$RESULTS/campaign_plan.json"
cp "$REPO/condor/pixi-matched-multiclass.toml" "$REPO/pixi.toml"
cp "$REPO/condor/pixi-matched-multiclass.lock" "$REPO/pixi.lock"
export PYTHONPATH="$REPO/src:$REPO/scripts"
STAGE=environment_setup
if command -v pixi >/dev/null 2>&1; then
  PIXI=$(command -v pixi)
else
  curl --retry 5 --retry-all-errors --retry-delay 5 -fsSL https://pixi.sh/install.sh -o "$WORK_DIR/install-pixi.sh"
  for attempt in 1 2 3; do
    if PIXI_NO_PATH_UPDATE=1 bash "$WORK_DIR/install-pixi.sh" && [[ -x "$PIXI_HOME/bin/pixi" ]]; then break; fi
    [[ "$attempt" != 3 ]] || { echo 'Pixi bootstrap failed'; exit 2; }
    sleep 10
  done
  PIXI="$PIXI_HOME/bin/pixi"
fi
cd "$REPO"
"$PIXI" install --manifest-path pixi.toml --platform linux-64 --locked 2>&1 | tee "$RESULTS/pixi_install.log"
RUN=(env "LD_LIBRARY_PATH=$REPO/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PIXI" run --manifest-path pixi.toml --platform linux-64 --locked)
cp pixi.lock "$RESULTS/pixi.lock.resolved"

if [[ "$MODE" == plot ]]; then
  STAGE=collect_verified_results
  timeout -k 30 1h "${RUN[@]}" python scripts/collect_hzz_4l_core_results.py \
    --results-prefix "$RESULT_PREFIX" --study-dir "$DATASET"
  STAGE=plot_matched_study
  "${RUN[@]}" python scripts/plot_hzz_4l_core_multiclass.py \
    --study-dir "$DATASET" --output-dir "$RESULTS/plots"
  STAGE=complete
  exit 0
fi

STAGE=verify_prepared_representation
fetch "$DATA/$DATA_KIND.tar.gz" "$WORK_DIR/$DATA_KIND.tar.gz" "$EXPECTED_DATA_SHA256"
extract "$WORK_DIR/$DATA_KIND.tar.gz" "$DATASET" "$DATA_KIND"
"${RUN[@]}" python - <<'PY' | tee "$RESULTS/runtime.txt"
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.__version__.startswith('2.11.0') and torch.version.cuda == '13.0'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
STAGE=download_verified_pretraining_source
SOURCE_ARGS=()
python3 - "$RUN_KEY" "$DATA_KIND" > "$WORK_DIR/source_files.tsv" <<'PY'
import json, re, sys
p = json.load(open('hzz_4l_multiclass_plan.json'))
r = p['runs'][sys.argv[1]]
if p['representations'][r['model_family']]['data_kind'] != sys.argv[2]:
    raise ValueError('Wrong representation for training run')
if r['condition'] != 'scratch':
    s = p['sources'][r['source_key']]
    for key, name in (('checkpoint', 'checkpoint.ckpt'), ('pretrain_config', 'config.yaml'),
                      ('pretrain_status', 'status.txt'), ('pretrain_completion', 'completion.json'),
                      ('pretraining_vocabulary_shard', 'vocabulary.parquet')):
        uri, digest = s.get(key), s.get(key + '_sha256')
        if key in ('pretrain_completion', 'pretraining_vocabulary_shard') and uri is None and digest is None:
            continue
        if not (uri and uri.startswith('gs://fcc-teststorage/') and not any(c.isspace() for c in uri)
                and digest and re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Unpinned training source: ' + key)
        print(uri, name, digest, sep='\t')
PY
if [[ -s "$WORK_DIR/source_files.tsv" ]]; then
  SOURCE_DIR="$WORK_DIR/pretrained"
  mkdir -p "$SOURCE_DIR"
  while IFS=$'\t' read -r uri name digest; do
    fetch "$uri" "$SOURCE_DIR/$name" "$digest"
  done < "$WORK_DIR/source_files.tsv"
  SOURCE_ARGS=(--source-dir "$SOURCE_DIR")
fi
STAGE=fit_and_held_out_evaluation
"${RUN[@]}" python scripts/hzz_4l_core_multiclass_study.py \
  --plan hzz_4l_multiclass_plan.json --dataset-root "$DATASET" --run-key "$RUN_KEY" \
  --membership-sha256 "$EXPECTED_MEMBERSHIP_SHA256" --output-dir "$RESULTS" "${SOURCE_ARGS[@]}" \
  2>&1 | tee "$RESULTS/training.log"
test -s "$RESULTS/completion.json"
test -s "$RESULTS/test_evaluation/test_predictions.npz"
test -s "$RESULTS/test_evaluation/evaluation_summary.json"
STAGE=complete
