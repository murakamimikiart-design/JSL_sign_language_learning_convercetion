#!/usr/bin/env python3
"""Local proxy: forwards /v1/messages -> api.anthropic.com"""
import http.server, urllib.request, urllib.error, json, sys

PORT = 8001
TARGET = 'https://api.anthropic.com'

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        # forward all x-api-key / anthropic-* headers
        fwd_headers = {
            'Content-Type': 'application/json',
            'anthropic-version': self.headers.get('anthropic-version', '2023-06-01'),
        }
        api_key = self.headers.get('x-api-key', '')
        if api_key:
            fwd_headers['x-api-key'] = api_key

        url = TARGET + self.path
        req = urllib.request.Request(url, data=body, headers=fwd_headers, method='POST')
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, x-api-key, anthropic-version, anthropic-dangerous-allow-browser')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

if __name__ == '__main__':
    server = http.server.HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f'Proxy running on port {PORT}')
    server.serve_forever()
