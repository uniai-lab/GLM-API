export CUDA_VISIBLE_DEVICES=1   # 0 8100 2 8200
export API_PORT=8200
export GLM_MODEL=glm-4-9b-chat
# export GLM_MODEL=DeepSeek-R1-Distill-Llama-8B

python3 ./api-v2.py