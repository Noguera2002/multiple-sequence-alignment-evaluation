#!/usr/bin/env bash
set -euo pipefail # set -euo pipefail ensures the script exits on any error

# Location assumptions: this script lives in src/data/scripts/
BASE_DIR="$(cd "$(dirname "$0")/../" && pwd)"           # src/data
RAW_DIR="$BASE_DIR/raw/RawSeqs"                        # input ungapped FASTAs
MSA_DIR="$BASE_DIR/msa"                                # output per-tool directories
VERSIONS_FILE="$MSA_DIR/versions.txt"               # file to log versions of MSA tools

# Ensure output directories exist
mkdir -p "$MSA_DIR/MAFFT"
mkdir -p "$MSA_DIR/Muscle"
mkdir -p "$MSA_DIR/Clustal"
mkdir -p "$MSA_DIR/T-coffee"

# Helper to log version if tool exists
log_version() {
  local name=$1 
  local cmd=$2
  if command -v $cmd >/dev/null 2>&1; then
    if output=$($cmd --version 2>&1); then
      echo "$name: $(echo "$output" | head -n1)" >> "$VERSIONS_FILE"
    else
      # Some tools use different flag to show version (e.g., muscle)
      if output=$($cmd -version 2>&1); then
        echo "$name: $(echo "$output" | head -n1)" >> "$VERSIONS_FILE"
      else
        echo "$name: version output unavailable" >> "$VERSIONS_FILE"
      fi
    fi
  else
    echo "$name: not found" >> "$VERSIONS_FILE"
  fi
}

# Clear previous versions summary and log available versions
echo "# MSA tool versions captured on $(date)" > "$VERSIONS_FILE"
log_version "MAFFT" "mafft"
log_version "Muscle" "muscle"
# Clustal Omega
if command -v clustalo >/dev/null 2>&1; then
  echo -n "Clustal Omega: " >> "$VERSIONS_FILE"
  clustalo --version | head -n1 >> "$VERSIONS_FILE"
else
  echo "Clustal Omega: not found" >> "$VERSIONS_FILE"
fi
# T-Coffee
log_version "T-Coffee" "t_coffee"

echo "Versions written to $VERSIONS_FILE"
echo

# Begin alignment loop
for rawf in "$RAW_DIR"/RF*.fa; do
  [ -e "$rawf" ] || continue
  fname=$(basename "$rawf")           # e.g., RF00001.fa
  fam=${fname%%.*}                    # RF00001

  echo "=== Family $fam ==="

  # MAFFT
  if command -v mafft >/dev/null 2>&1; then
    out="$MSA_DIR/MAFFT/${fam}_mafft.fasta"
    if [ -s "$out" ]; then
      echo "  MAFFT: exists, skipping"
    else
      echo "  Running MAFFT (default)..."
      mafft "$rawf" > "$out"
    fi
  else
    echo "  MAFFT: not installed, skipping"
  fi

  # Muscle
  if command -v muscle >/dev/null 2>&1; then
    out="$MSA_DIR/Muscle/${fam}_muscle.fasta"
    if [ -s "$out" ]; then
      echo "  Muscle: exists, skipping"
    else
      echo "  Running Muscle (default)..."
      muscle -in "$rawf" -out "$out"
    fi
  else
    echo "  Muscle: not installed, skipping"
  fi

  # Clustal Omega
  if command -v clustalo >/dev/null 2>&1; then
    out="$MSA_DIR/Clustal/${fam}_clustal.fasta"
    if [ -s "$out" ]; then
      echo "  Clustal Omega: exists, skipping"
    else
      echo "  Running Clustal Omega (default)..."
      clustalo -i "$rawf" -o "$out" --force --outfmt fa
    fi
  else
    echo "  Clustal Omega: not installed, skipping"
  fi

  # T-Coffee
  if command -v t_coffee >/dev/null 2>&1; then
    out="$MSA_DIR/T-coffee/${fam}_tcoffee.fasta"
    if [ -s "$out" ]; then
      echo "  T-Coffee: exists, skipping"
    else
      echo "  Running T-Coffee (default)..."
      t_coffee -infile "$rawf" -outfile "$out" -output fasta
    fi
  else
    echo "  T-Coffee: not installed, skipping"
  fi

  echo
done

echo "MSA generation complete."
