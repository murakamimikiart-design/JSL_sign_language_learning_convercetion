#!/bin/bash
cd "$(dirname "$0")"

# 既存プロセスを停止
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8001 | xargs kill -9 2>/dev/null
sleep 0.5

echo "================================"
echo "  JSL 手話認識アプリ 起動中..."
echo "================================"
echo ""
echo "  ブラウザが自動で開きます。"
echo "  このウィンドウは閉じないでください。"
echo ""
echo "  終了するときは赤い × ボタンで"
echo "  このウィンドウを閉じてください。"
echo "================================"
echo ""

# プロキシサーバーをバックグラウンドで起動
python3 proxy.py &
PROXY_PID=$!

# ブラウザを開く（2秒後）
(sleep 2 && open "http://localhost:8000/jsl_sign.html") &

# Ctrl+C を無効化してWebサーバーを起動
trap '' INT
python3 -m http.server 8000

# Webサーバー終了時にプロキシも停止
kill $PROXY_PID 2>/dev/null
