cd /home/xin.liang/code/VLM_gallery_creation/IDA-VLM/prepare_dataset/06_annotated_abcd

for f in benchmarks/*.json; do
  out="${f%.json}.jsonl"
  python prepare_jsonl.py --input "$f" --output "$out"
done