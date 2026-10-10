/* Actual Uvicorn + Streamlit + browser test. Only Fake/Mock suppliers. */
const {chromium}=require('playwright');
const {spawn,spawnSync}=require('child_process');
const fs=require('fs'),path=require('path'),os=require('os'),crypto=require('crypto');
const root=path.resolve(__dirname,'../..'),v2=path.join(root,'v2');
const python=process.env.V2_TEST_PYTHON || path.join(root,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'v2-browser-'));
const prefix='PersonalNotesReviewAssistant/Test/E2E/'+crypto.randomUUID();
const output=process.env.V2_UI_OUTPUT ? path.resolve(process.env.V2_UI_OUTPUT) : path.join(v2,'acceptance/phase19-20');
fs.mkdirSync(path.join(output,'screenshots'),{recursive:true});
const env={...process.env,PYTHONDONTWRITEBYTECODE:'1',PYTHONPATH:[v2,path.join(v2,'tests')].join(path.delimiter),
  V2_E2E_ROOT:temporary,V2_E2E_CREDENTIAL_PREFIX:prefix,V2_SETTINGS_TOKEN:'offline-e2e-nonce',V2_API_URL:'http://127.0.0.1:8019',V2_E2E_LONG_UI:'1'};
const results=[],servers=[],logs=[],layouts=[];
function server(label,args){
 const logfile=fs.openSync(path.join(output,label+'.log'),'w');logs.push(logfile);
 const process=spawn(python,['-B','-X','utf8',...args],{cwd:v2,env,windowsHide:true,stdio:['ignore',logfile,logfile]});servers.push(process);return process;
}
function apiServer(){return server('api',['-m','uvicorn','e2e_server:make_app','--factory','--host','127.0.0.1','--port','8019','--no-proxy-headers','--no-access-log']);}
async function waitHttp(url){for(let i=0;i<100;i++){try{let r=await fetch(url);if(r.ok)return;}catch{}await new Promise(r=>setTimeout(r,200));}throw Error('Service not ready: '+url);}
function assert(condition,message){if(!condition)throw Error(message);}
async function record(name,fn){await fn();results.push({name,status:'passed'});console.log('PASS '+name);}
async function main(){
 const fixtures=spawnSync(python,['-B','-X','utf8','-c',
  "import os,io,zipfile; from pathlib import Path; from PIL import Image; p=Path(os.environ['V2_E2E_ROOT']); (p/'browser.md').write_text('# 前端联调\\n## RRF\\nRRF 使用排名融合。\\n## BM25\\nBM25 使用词项匹配。\\n![配图](missing.png)',encoding='utf-8'); Image.new('RGB',(64,64),'white').save(p/'diagram.png'); (p/'invalid.png').write_bytes(b'not-image'); z=zipfile.ZipFile(p/'bundle.zip','w'); z.writestr('docs/note.md','# ZIP 学习\\n## 融合\\nRRF 使用排名。\\n![图](../images/diagram.png)\\n![缺失](C:/outside.png)'); z.write(p/'diagram.png','images/diagram.png'); z.close()"],{env,windowsHide:true,encoding:'utf8'});
 assert(fixtures.status===0,fixtures.stderr);
 let api=apiServer();server('frontend',['-m','streamlit','run',path.join(v2,'frontend/app.py'),'--server.address','127.0.0.1','--server.port','8519','--server.headless','true','--browser.gatherUsageStats','false']);
 await waitHttp('http://127.0.0.1:8019/health');await waitHttp('http://127.0.0.1:8519/_stcore/health');
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 const screenshot=async(name,focus)=>{
   await page.waitForTimeout(1000);
   await page.evaluate(()=>{window.scrollTo(0,0);for(const element of document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"]'))element.scrollTop=0;});
   await page.waitForTimeout(300);
   for(const size of [{width:1440,height:900},{width:1280,height:800}]){
     await page.setViewportSize(size);await page.waitForTimeout(250);
     if(focus)await page.locator(focus).evaluate(el=>el.scrollIntoView({block:'start'}));
     await page.screenshot({path:path.join(output,'screenshots',name+(size.width===1440?'':'-1280')+'.png'),fullPage:false});
     const geometry=await page.evaluate(()=>{
       const rect=selector=>{let element=document.querySelector(selector);if(!element)return null;let r=element.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};};
       return {horizontal_overflow:document.documentElement.scrollWidth>window.innerWidth,content:rect('.block-container'),navigation:rect('.st-key-top_navigation'),heading:rect('.page-heading'),deepseek:rect('.st-key-service_deepseek'),dashscope:rect('.st-key-service_dashscope'),balance:rect('.st-key-deepseek_balance_card')};
     });
     assert(!geometry.horizontal_overflow,'horizontal overflow '+name+' '+size.width);
     layouts.push({page:name,viewport:size,...geometry});
   }
   await page.setViewportSize({width:1440,height:900});
 };
 const link=async label=>{await page.getByRole('link',{name:label,exact:true}).click();await page.waitForTimeout(600);};
 const text=async value=>{await page.getByText(value,{exact:false}).first().waitFor({timeout:20000});};
 const upload=async(type,file)=>{
   await page.getByText(type,{exact:true}).click();
   await page.waitForTimeout(650);
   await page.locator('input[type=file]').setInputFiles(path.join(temporary,file));
   await page.waitForTimeout(650);
   const consent=page.getByRole('checkbox',{name:'我确认本次导入可能调用模型并产生费用'});
   if(!(await consent.isChecked()))await page.getByText('我确认本次导入可能调用模型并产生费用',{exact:true}).click();
   await page.waitForTimeout(650);
   await page.getByRole('button',{name:'确认导入',exact:true}).click();
 };
 try{
  await record('open_chat_empty_state_and_bottom_composer',async()=>{
    await page.goto('http://127.0.0.1:8519');await text('智能问答');await text('Fake 模型');await text('今天想理解哪个知识点');
    const history=await page.locator('.st-key-chat_history_panel').boundingBox();
    const input=await page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'}).boundingBox();
    assert(input.y>=history.y+history.height-5,'composer is not below chat history');
    await screenshot('01-chat-empty');
  });
  await record('knowledge_list_and_chapters',async()=>{await link('知识库');await text('章节目录');await text('BM25');});
  await record('markdown_text_only_upload_warning_and_no_image_switch',async()=>{
    assert(!(await page.getByRole('checkbox',{name:'识别 Markdown 引用的图片'}).count()),'old image switch visible');
    await upload('Markdown','browser.md');await text('导入成功');await text('本次仅导入文字');await screenshot('03-knowledge-markdown');
  });
  await record('repeat_upload_unchanged_no_auto_retry',async()=>{
    await page.getByRole('button',{name:'确认导入',exact:true}).click();await text('内容未变化，未重复入库');
  });
  await record('image_upload_over_http',async()=>{await upload('单张图片','diagram.png');await text('导入成功');});
  await record('invalid_image_error',async()=>{
    await page.locator('input[type=file]').setInputFiles(path.join(temporary,'invalid.png'));
    await page.getByText('invalid.png',{exact:true}).waitFor();await page.waitForTimeout(650);
    await page.getByRole('button',{name:'确认导入',exact:true}).click();await text('导入失败');
  });
  await record('zip_upload_partial_warning',async()=>{await upload('图文 ZIP','bundle.zip');await text('部分成功');await screenshot('02-knowledge-base');});
  await record('rag_sse_tokens_sources_and_done',async()=>{
    await link('智能问答');let input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});
    await input.fill('RRF 是什么？');await input.press('Enter');
    await page.getByText('▌',{exact:false}).first().waitFor({timeout:20000});
    assert(!(await page.getByText('回答完成',{exact:true}).count()),'draft arrived only after completion');
    await text('回答完成');await text('查看回答来源');await screenshot('01-chat');
    await page.getByText('查看回答来源',{exact:false}).click();await text('笔记引用');
    await screenshot('chat-with-source');
    await page.getByText('查看回答来源',{exact:false}).click();
  });
  await record('three_round_chat_alignment_sources_and_welcome_exit',async()=>{
    let expected=1;
    for(const question of ['BM25 和向量检索有什么区别？','怎样结合两种检索结果？']){
      expected++;
      const input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});
      await input.fill(question);await input.press('Enter');
      await page.waitForFunction(count=>document.querySelectorAll('[class*="st-key-chat_assistant_"]').length===count,expected);
      await page.waitForFunction(()=>{let t=document.querySelector('[data-testid="stChatInput"] textarea');return t&&!t.disabled&&!document.body.innerText.includes('▌');});
      await page.getByText('回答完成',{exact:true}).waitFor();await page.waitForTimeout(300);
    }
    assert(await page.locator('[class*="st-key-chat_user_"]').count()===3,'user history count');
    assert(await page.locator('[class*="st-key-chat_assistant_"]').count()===3,'assistant history count');
    assert(!(await page.getByText('今天想理解哪个知识点？',{exact:true}).count()),'welcome remains after question');
    const left=await page.locator('[class*="st-key-chat_assistant_"] [data-testid="stChatMessage"]').last().boundingBox();
    const right=await page.locator('[class*="st-key-chat_user_"] [data-testid="stChatMessage"]').last().boundingBox();
    assert(right.x>left.x+80,'role alignment absent');
    await screenshot('02-chat-three-rounds');
    await page.getByText('查看回答来源',{exact:false}).last().click();await page.getByText('笔记引用',{exact:true}).last().waitFor();
    await screenshot('02-chat-sources');
  });
  await record('five_round_chat_markdown_code_and_web_toggle',async()=>{
    const toggle=page.locator('.st-key-web_enabled input');
    await page.getByText('启用联网搜索',{exact:true}).click();await page.waitForFunction(()=>document.querySelector('.st-key-web_enabled input')?.checked===true);assert(await toggle.isChecked(),'web toggle did not turn on');
    let expected=3;
    for(const question of ['用代码演示阅读输入和输出。','再用一个小例子解释检索。']){
      expected++;
      const input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});
      await input.fill(question);await input.press('Enter');
      await page.waitForFunction(count=>document.querySelectorAll('[class*="st-key-chat_assistant_"]').length===count,expected);
      await page.waitForFunction(()=>{let t=document.querySelector('[data-testid="stChatInput"] textarea');return t&&!t.disabled&&!document.body.innerText.includes('▌');});
      await page.getByText('回答完成',{exact:true}).waitFor();await page.waitForTimeout(300);
    }
    assert(await page.locator('[class*="st-key-chat_assistant_"]').count()===5,'5 rounds not preserved');
    assert(await page.locator('[data-testid="stCode"]').count()>0,'Markdown code block missing');
    await screenshot('02-chat-five-rounds');
    await page.getByText('启用联网搜索',{exact:true}).click();await page.waitForFunction(()=>document.querySelector('.st-key-web_enabled input')?.checked===false);assert(!(await toggle.isChecked()),'web toggle did not turn off');
  });
  await record('sse_failure_discards_draft',async()=>{
    let input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});
    await input.fill('模拟失败');await input.press('Enter');await text('本轮没有保存助手回答');
  });
  await record('chat_history_navigation_and_clear',async()=>{
    await link('知识库');await link('智能问答');await text('RRF 是什么');
    await page.getByRole('button',{name:'清空当前聊天',exact:true}).click();await text('今天想理解哪个知识点');
  });
  await record('uncited_daily_web_only_and_previous_sources_boundaries',async()=>{
    const ask=async question=>{
      const expected=await page.locator('[class*="st-key-chat_assistant_"]').count()+1;
      const input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});
      await input.fill(question);await input.press('Enter');
      await page.waitForFunction(count=>document.querySelectorAll('[class*="st-key-chat_assistant_"]').length===count,expected);
      await page.waitForFunction(()=>{let t=document.querySelector('[data-testid="stChatInput"] textarea');return t&&!t.disabled&&!document.body.innerText.includes('▌');});
      await text('回答完成');await page.waitForTimeout(250);
    };
    await ask('RRF 是什么？');await ask('无引用测试：其他问题');
    assert(!(await page.locator('[class*="st-key-chat_assistant_"]').last().getByText('查看回答来源',{exact:false}).count()),'new answer reused prior sources');
    await page.getByRole('button',{name:'清空当前聊天',exact:true}).click();await text('今天想理解');
    await ask('无引用测试：其他问题');
    assert(!(await page.getByText('查看回答来源',{exact:false}).count()),'uncited chunks displayed as sources');
    await screenshot('chat-no-source');
    await page.getByRole('button',{name:'清空当前聊天',exact:true}).click();await text('今天想理解');await ask('你好');
    assert(!(await page.getByText('查看回答来源',{exact:false}).count()),'daily fabricated sources');
    await page.getByRole('button',{name:'清空当前聊天',exact:true}).click();await text('今天想理解');
    await page.getByText('启用联网搜索',{exact:true}).click();await ask('仅网页测试：RRF');
    await page.getByText('查看回答来源',{exact:false}).click();await text('网页引用');
    assert(!(await page.getByText('笔记引用',{exact:true}).count()),'web labelled as note');
    assert(!(await page.getByText('未引用网页夹具',{exact:false}).count()),'unused web displayed');
    await screenshot('chat-web-only');await page.getByText('启用联网搜索',{exact:true}).click();
  });
  await record('quiz_scope_and_count_controls',async()=>{
    await link('章节小测');await text('出题范围');await page.getByText('指定章节',{exact:true}).click();await text('选择专项章节');
    await page.getByText('整篇笔记',{exact:true}).click();
    const numbers=page.locator('input[type=number]');await numbers.nth(0).fill('1');await numbers.nth(0).press('Enter');await numbers.nth(1).fill('1');await numbers.nth(1).press('Enter');
    await screenshot('04-quiz-settings');
  });
  await record('quiz_generation_over_http',async()=>{
    await page.getByRole('button',{name:'生成小测',exact:true}).click();await text('实际生成 2 道');
    assert(!(await page.getByText('标准答案',{exact:false}).count()),'premature answer leakage');await screenshot('03-quiz');
  });
  await record('answer_navigation_preserves_drafts',async()=>{
    await page.getByText('B. 忽略原文中的限定条件',{exact:true}).click();
    await page.getByRole('button',{name:'下一题',exact:true}).click();
    await page.getByRole('textbox',{name:'用自己的话回答'}).fill('根据原笔记，用自己的语言表达概念及其作用。首次失败后测试重试。');
    await page.getByRole('button',{name:'上一题',exact:true}).click();
    assert(await page.getByRole('radio',{name:'B. 忽略原文中的限定条件'}).isChecked(),'choice lost on navigation');
    await page.getByRole('button',{name:'下一题',exact:true}).click();
    assert((await page.getByRole('textbox',{name:'用自己的话回答'}).inputValue()).includes('自己的语言'),'text draft lost');
  });
  await record('submit_grading_report_and_details',async()=>{
    await page.getByRole('button',{name:'提交小测',exact:true}).click();await text('部分题目批改失败');
    await page.getByRole('button',{name:'重试批改 / 报告',exact:true}).click();await text('学习报告');await text('本次重点巩固');
    const a=await page.locator('.st-key-report_strengthen').boundingBox(),b=await page.locator('.st-key-report_memory').boundingBox();
    assert(b.x>a.x+a.width,'report columns not distinct');
    assert(!(await page.locator('[data-testid="stExpander"] [data-testid="stExpander"]').count()),'nested report expanders');
    await screenshot('05-report-summary');
    await page.getByText(/^第 1 题 · 错误 ·/).click();await page.getByText(/^第 2 题 · 优秀 ·/).click();await text('标准答案');await text('参考答案');
    await screenshot('06-report-details','.st-key-report_details');
    await page.getByText(/^第 1 题 · 错误 ·/).click();await page.getByText(/^第 2 题 · 优秀 ·/).click();await screenshot('05-study-report');
  });
  await record('settings_first_use_and_mock_verification',async()=>{
    await link('设置');await text('模型服务设置');
    await screenshot('06-settings-first-use');
    const inputs=page.locator('input[type=password]');await inputs.nth(0).fill('offline-valid-deepseek');
    await page.getByText('同意本次 1 次验证调用及可能费用',{exact:true}).nth(0).click();
    await page.getByRole('button',{name:'验证并保存',exact:true}).nth(0).click();await text('已配置 · 服务端验证成功');
    await page.locator('input[type=password]').fill('offline-valid-dashscope');
    await page.getByText('同意本次 1 次验证调用及可能费用',{exact:true}).click();
    await page.getByRole('button',{name:'验证并保存',exact:true}).click();await text('••••cope');await screenshot('06-settings');
  });
  await record('balance_mock_refresh_success_failure_and_recovery',async()=>{
    assert(!(await page.getByText('¥12.50',{exact:true}).count()),'balance queried automatically');
    await page.getByRole('button',{name:'刷新余额',exact:true}).click();await text('¥12.50');await text('查询于');
    for(const hidden of ['充值余额','赠送余额','¥10.00','¥2.50'])assert(!(await page.getByText(hidden,{exact:true}).count()),'extra balance shown: '+hidden);
    await screenshot('06-settings');
    const cards=await page.evaluate(()=>{let a=document.querySelector('.st-key-service_deepseek').getBoundingClientRect(),b=document.querySelector('.st-key-service_dashscope').getBoundingClientRect();return {ay:a.y,by:b.y,ax:a.x,bx:b.x};});
    assert(Math.abs(cards.ay-cards.by)<3&&cards.bx>cards.ax,'Key cards not side by side');
    assert(await page.locator('.st-key-service_deepseek .st-key-deepseek_balance_card').count()===1,'balance not inside DeepSeek card');
    assert(await page.locator('.st-key-service_dashscope').getByText('Mock Embedding',{exact:true}).count()===1,'actual embedding model missing');
    fs.writeFileSync(path.join(temporary,'balance-failure'),'offline');
    await page.getByRole('button',{name:'刷新余额',exact:true}).click();await text('余额查询失败');
    assert(!(await page.getByText('¥12.50',{exact:true}).count()),'failed balance retained stale amount');
    fs.unlinkSync(path.join(temporary,'balance-failure'));
    await page.getByRole('button',{name:'刷新余额',exact:true}).click();await text('¥12.50');
  });
  await record('failed_key_replace_preserves_valid_key',async()=>{
    await page.getByRole('button',{name:'更换 Key',exact:true}).nth(0).click();
    await page.locator('input[type=password]').fill('bad-offline-key');
    // Existing consent state can be retained after changing the key.
    const consent=page.getByRole('checkbox',{name:'同意本次 1 次验证调用及可能费用'});
    if(!(await consent.isChecked()))await page.getByText('同意本次 1 次验证调用及可能费用',{exact:true}).click();
    await page.getByRole('button',{name:'验证并保存',exact:true}).click();await text('原有效配置已保留');
    const response=await fetch('http://127.0.0.1:8019/settings/providers',{headers:{'X-V2-Settings-Token':'offline-e2e-nonce'}});
    const data=(await response.json()).data;assert(data.deepseek.masked_key.endsWith('seek'),'failed replacement changed key');
  });
  await record('dashscope_replace_and_failed_validation_keep_saved_key',async()=>{
    await page.getByRole('button',{name:'更换 Key',exact:true}).nth(1).click();
    await page.locator('.st-key-service_dashscope input[type=password]').fill('offline-changed-scope');
    await page.locator('.st-key-service_dashscope').getByText('同意本次 1 次验证调用及可能费用',{exact:true}).click();
    await page.locator('.st-key-service_dashscope').getByRole('button',{name:'验证并保存',exact:true}).click();await text('••••cope');
    await page.getByRole('button',{name:'更换 Key',exact:true}).nth(1).click();
    await page.locator('.st-key-service_dashscope input[type=password]').fill('bad-dashscope-candidate');
    const consent=page.locator('.st-key-service_dashscope').getByRole('checkbox');
    if(!(await consent.isChecked()))await consent.check();
    await page.locator('.st-key-service_dashscope').getByRole('button',{name:'验证并保存',exact:true}).click();await text('原有效配置已保留');
    const response=await fetch('http://127.0.0.1:8019/settings/providers',{headers:{'X-V2-Settings-Token':'offline-e2e-nonce'}});
    const state=(await response.json()).data.dashscope;
    assert(state.source==='windows_credentials'&&state.validation_state==='verified','failed replacement damaged DashScope key');
  });
  await record('delete_key_confirmation_and_dashscope_environment_fallback',async()=>{
    fs.writeFileSync(path.join(temporary,'dashscope-env-fallback'),'Mock only');
    await page.getByRole('button',{name:'删除配置',exact:true}).nth(1).click();await text('确定删除吗');
    await page.getByRole('button',{name:'确认删除',exact:true}).click();await text('已配置（.env）· 未验证');
    const response=await fetch('http://127.0.0.1:8019/settings/providers',{headers:{'X-V2-Settings-Token':'offline-e2e-nonce'}});
    const state=(await response.json()).data.dashscope;
    assert(state.configured&&state.source==='environment'&&state.validation_state==='not_verified','incorrect fallback state');
    await screenshot('06-settings-env-fallback');
  });
  await record('refresh_recovers_key_configuration_and_no_secret_storage',async()=>{
    await page.reload();await text('服务端验证成功');
    const stores=await page.evaluate(()=>JSON.stringify({local:{...localStorage},session:{...sessionStorage}}));
    assert(!stores.includes('offline-valid')&&!stores.includes('bad-offline'),'secret found in browser storage');
  });
  await record('long_chapters_search_fold_and_scroll',async()=>{
    await link('知识库');
    await page.getByRole('combobox',{name:'选择笔记',exact:true}).click();
    await page.getByRole('option',{name:/长目录联调笔记/}).click();await text('章节 00');
    const tree=page.locator('.st-key-chapter_tree');
    const scroll=await tree.evaluate(el=>{let elements=[el,...el.querySelectorAll('*')];return elements.some(e=>e.scrollHeight>e.clientHeight+5&&['auto','scroll'].includes(getComputedStyle(e).overflowY));});
    assert(scroll,'long chapter tree has no internal scroll');
    assert((await tree.boundingBox()).height<=320,'chapter tree exceeds height bound');
    await screenshot('07-long-chapters');
    await page.getByRole('textbox',{name:'搜索章节',exact:true}).fill('章节 79');await page.getByRole('textbox',{name:'搜索章节',exact:true}).press('Enter');
    await text('章节 79');await page.getByText('章节 00',{exact:true}).waitFor({state:'hidden',timeout:15000});
    await page.getByText('长目录',{exact:true}).click();
    await page.getByText('知识点',{exact:true}).waitFor({state:'hidden',timeout:10000});
  });
  await record('long_report_extracts_expand_to_full_content_and_real_omissions',async()=>{
    await link('章节小测');
    if(await page.getByRole('button',{name:'重新设置小测',exact:true}).count())await page.getByRole('button',{name:'重新设置小测',exact:true}).click();
    await text('出题范围');
    await page.getByRole('combobox',{name:'选择学习笔记',exact:true}).click();
    await page.getByRole('option',{name:/长目录联调笔记/}).click();
    const numbers=page.locator('input[type=number]');await numbers.nth(0).fill('1');await numbers.nth(0).press('Enter');await numbers.nth(1).fill('1');await numbers.nth(1).press('Enter');
    await page.getByRole('button',{name:'生成小测',exact:true}).click();await text('实际生成 2 道');
    await page.getByText('B. 忽略原文中的限定条件',{exact:true}).click();await page.getByRole('button',{name:'下一题',exact:true}).click();
    await page.getByRole('textbox',{name:'用自己的话回答'}).fill('这是离线长报告布局验证，理解输入、处理和输出，但遗漏条件。');
    await page.getByRole('textbox',{name:'用自己的话回答'}).press('Tab');
    await page.getByRole('button',{name:'提交小测',exact:true}).click();await text('核心知识 0');
    await text('遗漏原文中的限定条件');await text('正确理解');
    const strengthen=await page.locator('.st-key-report_strengthen').boundingBox(),core=await page.locator('.st-key-report_core').boundingBox();
    assert(strengthen.y<core.y,'actual mistakes are not first');
    assert(!(await page.getByText('分清输入、处理和输出。'.repeat(22),{exact:true}).count()),'long explanation not collapsed');
    await screenshot('08-long-report');
    assert(await page.locator('.st-key-report_core').getByText(/核心知识/).count()===3,'more than 3 core points');
    assert(await page.locator('.st-key-report_memory').getByText(/记法/).count()===1,'more than 1 memory method');
    await page.getByText('完整学习报告',{exact:true}).click();await text('分清输入、处理和输出。'.repeat(22));
    await text('全部核心考点');await text('核心知识 6');await text('记法 3');
    assert(!(await page.locator('[data-testid="stExpander"] [data-testid="stExpander"]').count()),'nested full report');
    await screenshot('report-full-expanded','.st-key-report_full');
    await page.getByText('完整学习报告',{exact:true}).click();
    await page.getByText(/^第 2 题 · 良好 ·/).click();await text('改进建议');await text('遗漏点');
    await screenshot('09-long-report-details','.st-key-report_details');
    await page.getByText(/^第 2 题 · 良好 ·/).click();
  });
  for(const short of [0,1])await record(short?'all_correct_report_has_no_actual_errors':'choice_only_report_rating_not_applicable',async()=>{
    await page.getByRole('button',{name:'再做一次小测',exact:true}).click();await text('出题范围');
    await page.getByRole('combobox',{name:'选择学习笔记',exact:true}).click();await page.getByRole('option',{name:/^联调学习笔记/}).click();
    const numbers=page.locator('input[type=number]');await numbers.nth(0).fill('1');await numbers.nth(0).press('Enter');await numbers.nth(1).fill(String(short));await numbers.nth(1).press('Enter');
    await page.getByRole('button',{name:'生成小测',exact:true}).click();await text(`实际生成 ${1+short} 道`);
    await page.getByText('A. 依据原文理解术语关系',{exact:true}).click();
    if(short){await page.getByRole('button',{name:'下一题',exact:true}).click();await page.getByRole('textbox',{name:'用自己的话回答'}).fill('正确理解原笔记概念并用自己的话表达。');await page.getByRole('textbox',{name:'用自己的话回答'}).press('Tab');}
    await page.getByRole('button',{name:'提交小测',exact:true}).click();await text('本次没有记录到实际错误');
    if(!short)await text('不适用');else await text('优秀');
    assert(!(await page.locator('.st-key-report_strengthen').getByText('问题：',{exact:false}).count()),'fabricated actual error');
    await screenshot(short?'report-all-correct':'report-choice-only');
  });
  await record('actual_empty_collection_sources_and_import_prompt',async()=>{
    fs.writeFileSync(path.join(temporary,'empty-library'),'offline');api.kill();await new Promise(r=>setTimeout(r,600));api=apiServer();await waitHttp('http://127.0.0.1:8019/health');
    assert((await (await fetch('http://127.0.0.1:8019/notes')).json()).data.length===0,'fixture library not actually empty');
    await link('智能问答');await page.getByRole('button',{name:'清空当前聊天',exact:true}).click();await text('今天想理解');
    let input=page.getByRole('textbox',{name:'输入你的问题，例如：这两个概念有什么区别？'});await input.fill('RRF 是什么？');await input.press('Enter');await text('请先到知识库导入相关笔记');await text('回答完成');
    assert(!(await page.getByText('查看回答来源',{exact:false}).count()),'empty library fabricated sources');
    await screenshot('chat-empty-library');fs.unlinkSync(path.join(temporary,'empty-library'));
  });
  await record('backend_down_frontend_error_then_restart_state',async()=>{
    const old=(await (await fetch('http://127.0.0.1:8019/health')).json()).data.instance_id;
    api.kill();await new Promise(r=>setTimeout(r,600));
    await link('章节小测');await text('无法连接后端');
    api=apiServer();await waitHttp('http://127.0.0.1:8019/health');
    const next=(await (await fetch('http://127.0.0.1:8019/health')).json()).data.instance_id;assert(old!==next,'backend did not restart');
    await link('章节小测');await text('后端已重启');await text('出题范围');
    await link('设置');await text('服务端验证成功');
  });
  await record('1440_layout_and_no_browser_exceptions',async()=>{
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'horizontal overflow');
    assert(errors.length===0,'browser errors: '+errors.join(','));
  });
 }catch(error){
  await screenshot('failure');results.push({name:'failure',status:'failed',message:String(error)});
  throw error;
 }finally{
  await browser.close();
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
 for(const child of servers)child.kill();
 await new Promise(r=>setTimeout(r,300));
 for(const log of logs)fs.closeSync(log);
 spawnSync(python,['-B','-X','utf8','-c',"from app.api.credentials import WindowsCredentials; import os; s=WindowsCredentials(os.environ['V2_E2E_CREDENTIAL_PREFIX']); [s.delete(n) for n in ('deepseek','dashscope','disabled/deepseek','disabled/dashscope')]"],{cwd:v2,env,windowsHide:true});
 fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify({environment:'Fake Model + Mock HTTP + temporary Chroma',paid_requests:0,results,layouts},null,2));
});
