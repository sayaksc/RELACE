#!/usr/bin/env bash
# Install the text WebShop environment and prepare its 1,000-product catalog.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
WEBSHOP_DIR="${PROJECT_DIR}/agent_system/environments/env_package/webshop/webshop"
command -v java >/dev/null || { echo 'Install OpenJDK 11 before running this script.' >&2; exit 1; }
"$PYTHON_BIN" -m pip install -r "${PROJECT_DIR}/scripts/requirements-webshop.txt"
"$PYTHON_BIN" -m spacy download en_core_web_sm
mkdir -p "${WEBSHOP_DIR}/data"
cd "${WEBSHOP_DIR}/data"
if [[ ! -s items_shuffle_1000.json ]]; then
  "$PYTHON_BIN" -m gdown 'https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib' -O items_shuffle_1000.json
fi
if [[ ! -s items_ins_v2_1000.json ]]; then
  "$PYTHON_BIN" -m gdown 'https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu' -O items_ins_v2_1000.json
fi
if [[ ! -s items_human_ins.json ]]; then
  "$PYTHON_BIN" -m gdown 'https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O' -O items_human_ins.json
fi
cd "${WEBSHOP_DIR}/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k
"$PYTHON_BIN" convert_product_file_format.py
"$PYTHON_BIN" -m pyserini.index.lucene --collection JsonCollection --input resources \
  --index indexes --generator DefaultLuceneDocumentGenerator --threads 1 \
  --storePositions --storeDocvectors --storeRaw
"$PYTHON_BIN" -m pip check
