export CUDA_VISIBLE_DEVICES=3   # 0 8100 2 8200
export API_PORT=8200
# export GLM_MODEL=chatglm3-6b
# export GLM_MODEL=chatglm3-6b-128k
export GLM_MODEL=glm-4-9b-chat

python3 ./api-v2.py


# 提取关键信息：小王是一位软件工程师，他在一家科技公司工作了五年，主要负责开发和维护公司的核心系统
# 帮我生成一首中文的诗，不少于2000字
