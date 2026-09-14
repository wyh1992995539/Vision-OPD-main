"""Build the retrospective's standalone reader. Requires Python Markdown.
Usage: python docs/reader_tools/build_reader.py
Pinned JS assets are downloaded on first use, then cached beside this script.
"""
from pathlib import Path
import hashlib
import html
import re
import shutil
import urllib.request
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'project_retrospective.md'
OUTPUT = SOURCE.with_suffix('.html')
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
for name, url in [('mathjax.js', 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js'), ('mermaid.js', 'https://cdn.jsdelivr.net/npm/mermaid@10.9.3/dist/mermaid.min.js')]:
    target = ASSETS / name
    if not target.exists():
        cached = Path('/tmp/retrospective-reader-assets') / name
        if cached.exists():
            shutil.copyfile(cached, target)
        else:
            target.write_bytes(urllib.request.urlopen(url, timeout=60).read())

class MathProtection(Preprocessor):
    def run(self, lines):
        text = '\n'.join(lines)
        pattern = re.compile(r'(?P<fence>^```[^\n]*\n.*?^```[^\n]*$)|(?P<code>`+[^`\n]+`+)|(?P<display>\$\$[\s\S]+?\$\$)|(?P<inline>(?<![\\$])\$(?!\$)[^\n$]+?(?<!\\)\$)', re.M)
        def replace(m):
            if m.group('fence') or m.group('code'):
                return m.group()
            display = bool(m.group('display'))
            tex = m.group()[2:-2] if display else m.group()[1:-1]
            tag = 'div' if display else 'span'
            delimiters = ('\\[', '\\]') if display else ('\\(', '\\)')
            rendered = f'<{tag} class="math-{ "display" if display else "inline" }">{delimiters[0]}{html.escape(tex)}{delimiters[1]}</{tag}>'
            token = self.md.htmlStash.store(rendered)
            return '\n\n' + token + '\n\n' if display else token
        return pattern.sub(replace, text).split('\n')
class MathExtension(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.register(MathProtection(md), 'preserve_math', 29)

raw = SOURCE.read_text()
md = markdown.Markdown(extensions=[MathExtension(), 'extra', 'sane_lists', 'toc'], extension_configs={'toc': {'toc_depth': '2-4', 'permalink': False}})
body = md.convert(raw)
body = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', r'<pre class="mermaid">\1</pre>', body)
body = re.sub(r'(<table>[\s\S]*?</table>)', r'<div class="table-scroll" tabindex="0">\1</div>', body)
css = '''
:root{color-scheme:light;--ink:#233044;--muted:#67758a;--accent:#176b73;--line:#dce3e9;scroll-behavior:smooth}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
.sidebar{position:fixed;inset:0 auto 0 0;width:300px;background:#f8fafb;border-right:1px solid var(--line);padding:26px 22px;overflow:auto}.brand{font-size:20px;font-weight:750;color:var(--accent)}.caption{font-size:12px;color:var(--muted)}input{width:100%;padding:10px;margin:18px 0;border:1px solid var(--line);border-radius:8px;font:inherit;background:white}.toc ul{list-style:none;padding-left:13px}.toc>ul{padding:0}.toc li{margin:8px 0;line-height:1.5}.toc a{font-size:13px;color:#526078;text-decoration:none}.toc a:hover,.toc a.active{color:var(--accent);font-weight:650}
main{margin-left:300px;padding:32px 4vw 90px}article{max-width:1040px;margin:auto;background:white;padding:45px 55px;border:1px solid var(--line);border-radius:12px;box-shadow:0 5px 28px #23304405}.toolbar{max-width:1040px;margin:0 auto 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar span{color:var(--muted);font-size:13px}button,.source-link{padding:7px 13px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);font:inherit;font-size:13px;cursor:pointer;text-decoration:none}a{color:var(--accent);overflow-wrap:anywhere}h1{font-size:32px;line-height:1.35;margin:0 0 28px}h2{font-size:25px;margin-top:60px;padding-bottom:12px;border-bottom:2px solid #dce9e8}h3{font-size:21px;margin-top:38px}h4{font-size:18px;margin-top:28px}h5{font-size:16px}h1,h2,h3,h4,h5{scroll-margin-top:24px;line-height:1.5}p{margin:16px 0}li{margin:6px 0}blockquote{margin:20px 0;padding:2px 20px;border-left:4px solid #8bb6b4;background:#f3f8f8;color:#536477}pre{overflow:auto;background:#f3f5f8;padding:18px;border-radius:8px;line-height:1.6;font-size:13px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;background:#f0f3f6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre code{padding:0;background:transparent;font-size:inherit}.table-scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid var(--line);padding:11px 14px;text-align:left;min-width:110px}th{background:#edf4f4}tr:nth-child(even){background:#fafbfc}.math-display{overflow-x:auto;margin:24px 0;padding:12px 18px;background:#fafbfe;border:1px solid #e8edf3;border-radius:8px;font-size:1.08em}mjx-container[jax="SVG"]{max-width:none}mjx-container[jax="SVG"][display="true"]{margin:.5em 0!important}.mermaid{background:white;text-align:center;overflow:auto}.mermaid svg{height:auto;max-width:100%}img{max-width:100%}.footnote{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:24px;margin-top:50px}.mobile-toggle{display:none}
@media(max-width:1000px){.sidebar{width:250px}main{margin-left:250px;padding:22px}article{padding:30px}}@media(max-width:720px){.sidebar{display:none;z-index:10;width:85vw;box-shadow:4px 0 20px #0002}.sidebar.open{display:block}main{margin-left:0;padding:15px}article{padding:24px 18px}h1{font-size:26px}.mobile-toggle{display:inline-block}body{font-size:15px}}
@media print{.sidebar,.toolbar{display:none}main{margin:0;padding:0}article{max-width:none;border:0;box-shadow:none;padding:0}body{font-size:10pt;background:white}h2,h3,h4{break-after:avoid}pre{white-space:pre-wrap}table{font-size:8pt}th,td{min-width:0;padding:5px}.table-scroll,.math-display{overflow:visible}a{color:inherit}h2{margin-top:28px}@page{size:A4;margin:16mm}}
'''
mathjax = (ASSETS / 'mathjax.js').read_text().replace('</script', '<\\/script')
mermaid = (ASSETS / 'mermaid.js').read_text().replace('</script', '<\\/script')
config = r'''window.MathJax={loader:{load:[]},tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']],packages:{'[-]':['autoload','require']}},svg:{fontCache:'local'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};'''
# JS literals require two backslashes to encode one TeX delimiter.
interaction = '''
const status=document.getElementById('status');
Promise.all([MathJax.startup.promise,mermaid.run({querySelector:'.mermaid'})]).then(()=>{const errors=document.querySelectorAll('[data-mjx-error]');status.textContent=errors.length?'部分公式需要检查':'公式与图表已渲染 · 可离线阅读';}).catch(e=>{status.textContent='部分公式或图表渲染失败';console.error(e)});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();document.querySelectorAll('.toc li').forEach(li=>{li.style.display=li.textContent.toLowerCase().includes(q)?'':'none'})});
document.getElementById('print').onclick=()=>window.print();document.getElementById('menu').onclick=()=>document.querySelector('.sidebar').classList.toggle('open');
document.querySelectorAll('.toc a').forEach(a=>a.addEventListener('click',()=>{document.querySelectorAll('.toc a.active').forEach(x=>x.classList.remove('active'));a.classList.add('active');document.querySelector('.sidebar').classList.remove('open')}));
'''
page = '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vision-OPD 项目复盘 · 阅读版</title><style>'+css+'</style></head><body>'
page += '<aside class="sidebar"><div class="brand">Vision-OPD</div><div class="caption">项目复盘 · 章节导航</div><input id="search" type="search" aria-label="筛选章节" placeholder="筛选章节…"><nav aria-label="目录">'+md.toc+'</nav></aside>'
page += '<main><div class="toolbar"><button id="menu" class="mobile-toggle">目录</button><button id="print">打印 / 保存 PDF</button><a class="source-link" href="project_retrospective.md">Markdown 原稿</a><span id="status" role="status">正在渲染公式与图表…</span></div><article>'+body
page += '<p class="footnote">由 Markdown 原稿生成。内嵌 MathJax 3.2.2 与 Mermaid 10.9.3；阅读版为生成时的快照。<br>原稿 SHA256：'+hashlib.sha256(raw.encode()).hexdigest()+'</p></article></main>'
page += '<script>'+config+'</script><script>'+mathjax+'</script><script>'+mermaid+'</script><script>mermaid.initialize({startOnLoad:false,securityLevel:"strict",theme:"neutral",maxTextSize:100000});'+interaction+'</script></body></html>'
OUTPUT.write_text(page)
print(f'Generated: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)')
print(f'Math: {body.count(chr(34)+"math-display"+chr(34))} display, {body.count(chr(34)+"math-inline"+chr(34))} inline; diagrams: {body.count(chr(34)+"mermaid"+chr(34))}')
