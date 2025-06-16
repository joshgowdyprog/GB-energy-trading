#!/bin/bash

output_file="large_blobs_report.txt"
echo "Large blob filename mapping started at $(date)" > "$output_file"

blobs=(
  2f351380409b6ba64d6a827304a25f92bcac57bc
  5f5f1b9761ea1b3f0b20b600ed6db445d46f5141
  548d039c6595d485dcd0fc320419dcc391a19948
  79d8c7ec40fbd67c0ee459eade2a926fd29de674
  660b67a34ad59ea6c0a467e9f484df9debe5bb8b
  856bd003d9793e7d276c8207105f96511a43b455
  c07431316fcdf41cad859215ff6b55a4ad72c3c9
  770c1bb94e62a8ed35e90eacfa6e306450335d89
  527987e92727551adcf87222d27a305efdae63d1
  f97d28ee1e1a12098ceaee71666d1b9de4a0d5e7
  53a398e2ffb77cbb6d6c7f343a594a86fb410509
  310d24b2006897a828d7266c19922280bfa0d5bd
  e7780b44e8fd653d60da48b2d796384384ceb9c4
  576c9eb01d46186d75c721aa69ca41edffd3281b
  628c9e7f76984a9a1b1ee7e1c708804b46c5b60d
  6d9fdcf7e46d7883451e88ae890dbc8dea67e00e
  31c1f4a65894a53dcff84f9cff31c5826c38eff0
  2af64da9b872834bc69734fa6b58610ab0ff3f38
  c04be4c9c9fce2a32c4c94b69a27dbefe874ffa7
  623bd1a92a370eabd3f66038052d931d9418191e
)

echo "Searching for filenames corresponding to large blobs..."

for blob in "${blobs[@]}"; do
  echo -e "\n=== Blob: $blob ===" | tee -a "$output_file"
  found=false
  while read commit; do
    result=$(git ls-tree -r "$commit" | grep "$blob")
    if [[ $result ]]; then
      found=true
      echo "Found in commit: $commit" | tee -a "$output_file"
      echo "$result" | tee -a "$output_file"
    fi
  done < <(git rev-list --all)
  if [ "$found" = false ]; then
    echo "No file match found in any commit." | tee -a "$output_file"
  fi
done

echo -e "\nDone. Report saved to $output_file"