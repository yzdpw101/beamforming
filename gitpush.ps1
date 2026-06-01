# beamforming 项目 Git 快捷操作
# 用法: .\gitpush.ps1 "提交信息"

param([string]$msg = "update")

git add -A
git commit -m "$msg"
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main
