# beamforming 项目 Git 快捷操作
param([string]$msg = "update")

git add -A
git commit -m "$msg"
# 先试代理，失败走直连
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main 2>$null
if ($LASTEXITCODE -ne 0) { git push origin main }
