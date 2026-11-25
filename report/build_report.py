"""Build editable reports from measured outputs. Requires bundled python-docx.
Regeneration replaces both reports; preserve any manually edited Word copy first.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import configuration_fingerprint
COVER_NAMES=['Aditya Vilas','Sameet Vilas','Jay Manohar Kawade','Raghav Goenka','Pragati Garg','Saksham Mishra']
TITLE='IDO AUM Lite'
SUBTITLE='Wrong events and area under margin\nfor learning with noisy labels'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def h(text):return {'kind':'heading','text':text}
def p(text):return {'kind':'paragraph','text':text}
def eq(name):return {'kind':'equation','name':name}
def table(caption,headers,rows,widths):return dict(kind='table',caption=caption,headers=headers,rows=rows,widths=widths)
def fig(name,caption,width=6.65):return dict(kind='figure',name=name,caption=caption,width=width)
def inputs():
 root=ROOT/'results';m=json.loads((root/'metrics.json').read_text());receipt=json.loads((root/'metrics_manifest.json').read_text())
 for name,key in [('metrics.json','metrics_json_sha256'),('metrics.csv','metrics_csv_sha256')]:assert digest(root/name)==receipt[key]
 cfg=json.loads((root/'experiment_config.json').read_text())
 assert cfg==m['scope']['configuration'] and configuration_fingerprint(cfg)==m['scope']['configuration_sha256']
 assert m['complete'] and m['row_count']==6
 plots=json.loads((root/'plots_manifest.json').read_text())
 assert plots['sources']['metrics_json_sha256']==digest(root/'metrics.json')
 for name,sha in plots['artifacts'].items():assert digest(root/name)==sha
 return cfg,m

def content(cfg,m):
 rows={(x['row']['noise_ratio'],x['row']['method']):x for x in m['methods']}
 noises=cfg['noise_ratios'];lo,hi=noises;W,C,N,T=[cfg[k] for k in ['warmup_epochs','continuation_epochs','train_size','test_size']]
 def acc(r,method):return rows[r,method]['row']['test_accuracy_pct']
 def gain(r,method):return acc(r,method)-acc(r,'standard')
 detectors={x['noise_ratio']:x for x in m['detectors']};env=m['methods'][0]['evaluation']['environment']
 seconds=sum(rows[r,'standard']['warmup_training'][k] for r in noises for k in ['training_seconds_excluding_checkpoint','checkpoint_publication_seconds'])+sum(x['continuation_training'][k] for x in m['methods'] for k in ['training_seconds_excluding_checkpoint','checkpoint_publication_seconds'])
 skips=sum(rows[r,'standard']['warmup_training']['amp_skipped_steps'] for r in noises)+sum(x['continuation_training']['amp_skipped_steps'] for x in m['methods'])
 accuracy=[[f'{r:.0%}',label,f'{acc(r,method):.2f}',f'{gain(r,method):+.2f}'] for r in noises for method,label in [('standard','Standard'),('ido_lite','IDO-lite'),('hybrid','Hybrid')]]
 aucs=[[f'{r:.0%}',*[f'{detectors[r]["primary"][k]:.6f}' for k in ['we_roc_auc','negative_aum_roc_auc','hybrid_roc_auc']]] for r in noises]
 diagnostics=[]
 for r in noises:
  for method,label in [('ido_lite','IDO-lite'),('hybrid','Hybrid')]:
   d=rows[r,method]['continuation_training']['sample_weighted_diagnostics']
   assert abs(d['supervised_loss']+d['pseudo_loss']+d['consistency_loss']-d['total_loss'])<1e-7
   diagnostics.append([f'{r:.0%}',label,f'{100*d["confidence_fraction"]:.2f}',*[f'{d[k]:.4f}' for k in ['supervised_loss','pseudo_loss','consistency_loss','total_loss']]])
 pages=[]
 pages.append([
 h('Abstract'),p(f'This study evaluates whether combining Wrong Events (WE) and Area Under the Margin (AUM) improves learning with noisy labels. The experiment uses {N:,} CIFAR-10 training images, exact 20% and 40% label flips, seed {cfg["seed"]}, and a shared {W}-epoch warm-up followed by {C}-epoch continuations. IDO-lite reaches {acc(lo,"ido_lite"):.2f}% and {acc(hi,"ido_lite"):.2f}% test accuracy, versus {acc(lo,"standard"):.2f}% and {acc(hi,"standard"):.2f}% for Standard cross-entropy. Hybrid reaches {acc(lo,"hybrid"):.2f}% and {acc(hi,"hybrid"):.2f}%. AUM detects corrupted labels more effectively than WE, but fixed 50:50 fusion does not improve classification over WE-only in this run. These are single-seed exploratory observations from an IDO-inspired simplified implementation, not a faithful reproduction or a statistically reliable superiority claim.'),
 h('1 Motivation and relation to prior work'),p('Wrong training labels and difficult correctly labelled images can both produce prediction disagreement. WE counts these events; a logit margin also records their direction and strength. The research questions are whether WE detects label corruption, how AUM compares, and whether a fixed blend improves final accuracy. This is a narrow multi-signal experiment, not an implementation of the earlier Group 2 proposal portfolio.'),
 p('Original IDO uses Wrong Events, class-wise beta mixture estimation and dynamic instance-level optimisation [1]. AUM ranks samples by average assigned-label margin and includes threshold-sample calibration [2]. This implementation keeps cumulative WE and raw-logit AUM, replaces mixture estimation with within-provided-label-class ranks, simplifies the adaptive objective, and omits original base-model selection and AUM threshold calibration. The established signals are not claimed as new.'),
 h('2 Data and fixed experimental design'),p(f'The same {N:,} original training IDs are used for both conditions, stratified by original class ({N//10:,} per class). The official {T:,}-image test split remains untouched [3]. Exactly {round(lo*N):,} or {round(hi*N):,} distinct subset positions are flipped uniformly to one of the nine other classes; every selected label changes. Selection uses seed {cfg["seed"]}, with independent recorded SeedSequence streams for the two corruption masks. Clean originals and masks are stored separately for analysis, never as score, loss or selection inputs.'),
 table('Table 1 Executed settings',['Item','Setting'],[
 ['Classifier','ResNet-18 from scratch; 3 by 3 stride-1 stem; no initial maxpool; 10 outputs [4]'],
 ['Optimiser',f'AdamW [5]; lr {cfg["learning_rate"]:g}; weight decay {cfg["weight_decay"]:g}; batch {cfg["batch_size"]}'],
 ['Schedule',f'{W} shared warm-up + {C} continuation epochs per final model; {2*(W+3*C)} executed training epochs'],
 ['Views','Crop 32x32 with padding 4 and horizontal flip; independent strong view adds jitter: brightness/contrast/saturation 0.2, hue 0.05'],
 ['Normalisation','Mean (0.4914, 0.4822, 0.4465); std (0.2470, 0.2435, 0.2616); deterministic statistics/test transforms'],
 ['Execution',f'{cfg["num_workers"]} workers; mixed precision enabled; constant learning rate; seed {cfg["seed"]}'],
 ['Evaluation','Fixed final checkpoints; entire clean test split; no test-set tuning']], [1.05,5.6]),
 p('Both adaptive branches share the same architecture, optimiser, augmentations, loss and refresh schedule; their intended difference is the suspicion score. Standard versus adaptive compares whole strategies. Numerical step skips are disclosed in Section 9.')])
 pages.append([
 h('3 Tracking and relative suspicion'),p('After each completed epoch, one deterministic eval/no-grad pass visits every stable subset ID exactly once. With supplied label y-tilde and raw logits z, WE counts epochs predicted wrongly, not correct-to-incorrect transitions. The indicator I is one when its condition holds.'),eq('we'),eq('margin'),eq('aum'),
 p('AUM averages margins over all completed observations, rather than using only the final margin. Within each provided-label class of size n, use one-based average ranks for ties. Rank larger WE and negative AUM as more suspicious. Empty classes are skipped; singleton or all-tied classes receive 0.5.'),eq('ranks'),eq('fusion'),eq('weights'),
 p(f'These are relative ranks, not corruption probabilities. Complete-class mean suspicion is 0.5 at both noise levels, imposing mean classification weight {cfg["clean_weight_floor"]+(1-cfg["clean_weight_floor"])*.5:.2f}. The {cfg["clean_weight_floor"]:g} floor and parabolic difficulty are experimental heuristics, not BMM estimates. Scores are frozen from prior completed epochs, then refreshed after the next full statistics pass. Standard continuation uses CE-only diagnostics without WE/AUM updates.'),
 h('4 Shared adaptive objective'),p(f'Weak and strong views produce probabilities p-w and p-s. A detached weak argmax gives pseudo-label q; mask m is one when detached weak confidence is at least {cfg["pseudo_threshold"]:g}. Both adaptive methods use the same weighted per-example terms:'),eq('supervised'),eq('pseudo'),eq('consistency'),eq('total'),
 p('All terms use the full-batch denominator, including masked examples and partial final batches. Pseudo targets and scores are detached; consistency differentiates both views. CE and softmax use float32 under autocast. Sample-weighted epoch means must sum to total. The classification floor prevents zero weights, not necessarily zero gradients.'),
 h('5 Fair branching and recovery'),p('Each continuation restores the complete shared warm-up state, including BatchNorm buffers, then resets AdamW, scaler, branch RNG streams and controlled loader generators. Adaptive statistics are independent copies. Epoch checkpoints preserve model/optimizer/scaler, statistics, warm-up history, logs, Python/NumPy/CPU/CUDA RNGs, loader generators and code/configuration/data identities. Verified publication retains a prior generation. Resume restores state after object construction and before the next batch; cross-device bitwise equality is not assumed.')])
 pages.append([
 h('6 Classification results'),p(f'All six fixed final models were evaluated once on the untouched {T:,}-image test split. Each has {W+C} training epochs in its history, while warm-up computation is shared once per noise level. Gains below are percentage points, not relative percentage improvements.'),
 table('Table 2 Final test accuracy',['Noise','Method','Accuracy percent','Gain vs Standard pp'],accuracy,[.65,1.3,2.1,2.6]),
 fig('accuracy_comparison.png','Figure 1 Measured final accuracy. Each bar is one seed; no across-seed uncertainty interval has been estimated.',6.4),
 p(f'IDO-lite exceeds Standard by {gain(lo,"ido_lite"):.2f}/{gain(hi,"ido_lite"):.2f} points at 20%/40% noise. Hybrid exceeds Standard by {gain(lo,"hybrid"):.2f}/{gain(hi,"hybrid"):.2f} points, but finishes {acc(lo,"ido_lite")-acc(lo,"hybrid"):.2f}/{acc(hi,"ido_lite")-acc(hi,"hybrid"):.2f} points below IDO-lite. The proposed fixed fusion is therefore not supported by this observed classification outcome. The Standard comparison includes changed loss and augmentation, so it does not isolate scoring.')])
 pages.append([
 h('7 Detection results and score distributions'),p(f'The primary detector result uses shared warm-up epoch {W}, with corrupted examples as positive. Earlier epochs are retrospective diagnostics, never selected by quality. Detector values repeat across continuation rows because their source warm-up is shared.'),
 table('Table 3 Primary detection ROC AUC',['Noise','WE','Negative AUM','Hybrid ranks'],aucs,[.65,1.7,2.1,2.2]),
 fig('noise_detection_roc.png','Figure 2 Raw WE versus negative-AUM ROC curves at the prespecified primary epoch. No learned cutoff or test-set tuning is applied.'),
 fig('score_distributions.png','Figure 3 Prespecified 20% noise condition. Groups are normalized separately; clean/corrupted labels are retrospective analysis only.'),
 p(f'AUM AUC is {detectors[lo]["primary"]["negative_aum_roc_auc"]:.4f}/{detectors[hi]["primary"]["negative_aum_roc_auc"]:.4f}, above WE at {detectors[lo]["primary"]["we_roc_auc"]:.4f}/{detectors[hi]["primary"]["we_roc_auc"]:.4f}. Warm-up WE has only {W+1} possible counts, whereas margins distinguish many ties. Hybrid AUC is just above AUM at 20% and below it at 40%. Score distributions overlap; neither detector produces a perfect partition or guarantees improved training weights.')])
 pages.append([
 h('8 Learning dynamics and loss diagnostics'),fig('common_ce_curves.png',f'Figure 4 Common deterministic unweighted noisy-label CE. The boundary follows epoch {W}; weighted adaptive objectives are not presented as CE.'),
 p('The shared warm-up is followed by independently reset continuations. Standard attains lower common noisy-label CE but lower test accuracy. Fitting the supplied labels better is not necessarily better generalisation when some supplied labels are wrong.'),
 table('Table 4 Sample weighted continuation averages',['Noise','Method','Confident percent','Supervised','Pseudo','Consistency','Total'],diagnostics,[.5,.85,1.0,1.05,.95,1.1,1.2]),
 p('Component columns include their weights. Confidence coverage is about 63% at 20% noise and 34-36% at 40%; pseudo-label supervision is active but often masked. Exact-zero pseudo terms are approximately 37%/37% and 66%/64% for WE-only/Hybrid at the two noise levels. Weighted supervised and total terms have no zero or near-zero examples at epsilon 1e-8. Consistency near-zero fractions span 0.00083-0.36417%. Mean c is about 0.55 and mean d spans 0.6801-0.6907. These are loss values, not gradient norms; coefficients do not equalise gradient influence.'),
 h('9 Runtime and numerical behaviour'),p(f'The saved environment records {env["gpu"]}, Python {env["python"]}, torch {env["torch"]}, torchvision {env["torchvision"]}, CUDA {env["cuda"]}. Unique saved training/inference and publication times sum to {seconds:.2f} seconds ({seconds/60:.2f} minutes), counting warm-up once per condition. This excludes setup/process and other untimed overhead, not a complete GPU-use audit. No fallback or threshold amendment was used. Work totals {2*(W+3*C)} training epochs, {2*(W+5*C)} image-view passes, {2*(W+2*C)} statistics passes, {2*C} CE-only passes and six final test passes.'),
 p(f'GradScaler recorded {skips} skipped optimizer updates: one in each Standard continuation and one in 40% Hybrid; none in warm-ups or WE-only. No extra epochs compensated for skips. Configured schedules match, but realised update counts differ slightly, which matters when interpreting small accuracy gaps.')])
 pages.append([
 h('10 Interpretation and limitations'),p('A stronger standalone detector need not create a better training weight. Both scores come from the same classifier and can share errors; class ranks discard absolute scale, and averaging can change the ordering used by all three loss roles. These are plausible explanations, not tested causal mechanisms. Supervised loss dominates the measured objective; no coefficient sweep was performed to seek a favourable result.'),
 p(f'This is one seed, one architecture, {N:,} images, {W+C}-epoch model histories and synthetic symmetric noise only. There is no across-seed variance estimate, significance test, hard-clean retention measurement, AUM-only training branch, real/asymmetric/instance-dependent noise evaluation or original-IDO reproduction. Coarse WE, rank-imposed mean suspicion, pseudo-label errors and the AMP skips limit inference. Passing implementation checks does not establish statistical reliability.'),
 h('11 Reproducibility and development'),p('The repository is github.com/Avilas01/ido-aum-lite. Training/evaluation used pinned commit 695f50f; the full configuration fingerprint is retained in results/experiment_config.json and the evaluation/plot manifests. Colab replication follows setup, hosted checks, benchmark, one frozen configuration, both noise conditions with --resume, and src.evaluate. Exact commands and the six-step recovery checklist are in README.md. Drive run-001 holds full checkpoints; results/ holds measured statistics and plots. The official archive is shared across sessions and extracted locally [6].'),
 p('The current implementation was developed with Codex assistance in VS Code on Windows. Work completed includes deterministic corruption, stable-ID tracking, shared losses, epoch recovery, notebook orchestration, cached evaluation and verified plots. The local lightweight suite passed 209 tests with six PyTorch-dependent modules skipped locally. A hosted-test command, integrated GPU smoke and completed-stage resume were observed successful; fresh-VM recovery of an interrupted real epoch was not demonstrated. No local torch/CUDA installation, CIFAR download or model training was used.'),
 h('12 Conclusion and future work'),p('Both adaptive methods exceed Standard in this run; the fixed hybrid does not exceed WE-only accuracy. AUM nevertheless gives stronger warm-up detection. Next steps are independent seeds, original-IDO and AUM-only ablations, structured/real noise and preregistered confidence/loss studies. These are future experiments, not completed evidence.'),
 h('References'),
 dict(kind='reference',text='[1] K. Zhang et al. Handling Label Noise via Instance-Level Difficulty Modeling and Dynamic Optimization. NeurIPS 38, 2025.',url='https://proceedings.neurips.cc/paper_files/paper/2025/hash/429e7b31625a8b7839f9e4d6e2aa9bb9-Abstract-Conference.html'),
 dict(kind='reference',text='[2] G. Pleiss, T. Zhang, E. Elenberg and K. Q. Weinberger. Identifying Mislabeled Data using the Area Under the Margin Ranking. NeurIPS 33, 2020.',url='https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html'),
 dict(kind='reference',text='[3] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. University of Toronto technical report, 2009.',url='https://www.cs.toronto.edu/~kriz/cifar.html'),
 dict(kind='reference',text='[4] K. He, X. Zhang, S. Ren and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.',url='https://arxiv.org/abs/1512.03385'),
 dict(kind='reference',text='[5] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019.',url='https://arxiv.org/abs/1711.05101'),
 dict(kind='reference',text='[6] Google. Colaboratory Frequently Asked Questions. Accessed November 2026.',url='https://research.google.com/colaboratory/faq.html')])
 return pages

def math_tree(name,cfg):
 from docx.oxml import OxmlElement as X
 from docx.oxml.ns import qn
 def r(text):
  node=X('m:r');props=X('w:rPr');fonts=X('w:rFonts')
  fonts.set(qn('w:ascii'),'Cambria Math');fonts.set(qn('w:hAnsi'),'Cambria Math');props.append(fonts)
  size=X('w:sz');size.set(qn('w:val'),'20');props.append(size);node.append(props)
  t=X('m:t');t.text=text;node.append(t);return node
 def seq(values):return [r(v) if isinstance(v,str) else copy.deepcopy(v) for v in values]
 def sub(base,index):
  node=X('m:sSub');e=X('m:e');e.extend(seq([base]));s=X('m:sub');s.extend(seq([index]));node.extend([e,s]);return node
 def frac(num,den):
  node=X('m:f');a=X('m:num');a.extend(seq(num));b=X('m:den');b.extend(seq(den));node.extend([a,b]);return node
 def summation(low,high,terms):
  node=X('m:nary');props=X('m:naryPr');c=X('m:chr');c.set(qn('m:val'),'∑');props.append(c);node.append(props)
  for key,values in [('sub',[low]),('sup',[high]),('e',terms)]:
   child=X('m:'+key);child.extend(seq(values));node.append(child)
  return node
 yi=sub('ỹ','i');wi=sub('WE','i');ai=sub('AUM','i');si=sub('s','i');ci=sub('c','i');di=sub('d','i')
 maps={
  'we':[wi,'(t) = ',summation('u=1','t',['I[',sub('arg max','k'),' ',sub('z','i'),'(u)[k] ≠ ',yi,']'])],
  'margin':[sub('δ','i'),'(t) = ',sub('z','i'),'(t)[',yi,'] − ',sub('max','k≠ỹᵢ'),' ',sub('z','i'),'(t)[k]'],
  'aum':[ai,'(t) = ',frac([summation('u=1','t',[sub('δ','i'),'(u)'])],['t'])],
  'ranks':[sub('s','W,i'),' = ',frac([sub('r','WE,i'),' − 0.5'],['n']),',   ',sub('s','A,i'),' = ',frac([sub('r','−AUM,i'),' − 0.5'],['n'])],
  'fusion':[si,' = ',sub('s','W,i'),'  (WE only);   ',si,' = ',frac([sub('s','W,i'),' + ',sub('s','A,i')],['2']),'  (Hybrid)'],
  'weights':[ci,f' = {cfg["clean_weight_floor"]:g} + {1-cfg["clean_weight_floor"]:g}(1 − ',si,');   ',di,' = 4',si,'(1 − ',si,')'],
  'supervised':[sub('S','i'),' = ',ci,' CE(',sub('z','w,i'),', ',yi,')'],
  'pseudo':[sub('P','i'),f' = {cfg["pseudo_coefficient"]:g}',si,sub('m','i'),' CE(',sub('z','s,i'),', ',sub('q','i'),')'],
  'total':['L = ',frac(['1'],['B']),' ',summation('i=1','B',['(',sub('S','i'),' + ',sub('P','i'),' + ',sub('K','i'),')'])]}
 difference=X('m:e');difference.extend(seq(['(',sub('p','w,i,k'),' − ',sub('p','s,i,k'),')']))
 square=X('m:sSup');square.append(difference);exponent=X('m:sup');exponent.append(r('2'));square.append(exponent)
 maps['consistency']=[sub('K','i'),f' = {cfg["consistency_coefficient"]:g}',di,' ',frac(['1'],['10']),' ',summation('k=1','10',[square])]
 omath=X('m:oMath');omath.extend(seq(maps[name]));return omath

MATH_MD={
 'we':r'WE_i(t)=\sum_{u=1}^{t}\mathbf{1}[\arg\max_k z_{i,k}(u)\ne\tilde y_i]',
 'margin':r'\delta_i(t)=z_{i,\tilde y_i}(t)-\max_{k\ne\tilde y_i}z_{i,k}(t)',
 'aum':r'AUM_i(t)=\frac{1}{t}\sum_{u=1}^{t}\delta_i(u)',
 'ranks':r's_{W,i}=(r_{WE,i}-0.5)/n,\qquad s_{A,i}=(r_{-AUM,i}-0.5)/n',
 'fusion':r's_i=s_{W,i}\ (WE\ only),\qquad s_i=(s_{W,i}+s_{A,i})/2\ (Hybrid)',
 'weights':r'c_i=0.1+0.9(1-s_i),\qquad d_i=4s_i(1-s_i)',
 'supervised':r'S_i=c_i CE(z_{w,i},\tilde y_i)',
 'pseudo':r'P_i=0.5s_i m_i CE(z_{s,i},q_i)',
 'consistency':r'K_i=0.5d_i\frac{1}{10}\sum_{k=1}^{10}(p_{w,i,k}-p_{s,i,k})^2',
 'total':r'L=\frac{1}{B}\sum_{i=1}^{B}(S_i+P_i+K_i)'}

def write_markdown(pages):
 lines=['# '+TITLE,'',SUBTITLE.replace('\n',' '),'','Project implementation and experimental evaluation','',*COVER_NAMES,'',
        'BITS Pilani, K. K. Birla Goa Campus','BITS F471 - Introduction to Large Language Models','Group 2','November 2026','']
 for page in pages:
  for block in page:
   kind=block['kind']
   if kind=='heading':lines+=['## '+block['text'],'']
   elif kind=='paragraph':lines+=[block['text'],'']
   elif kind=='equation':lines+=['$$',MATH_MD[block['name']],'$$','']
   elif kind=='figure':lines+=[f'![{block["caption"]}](../results/figures/{block["name"]})','',block['caption'],'']
   elif kind=='reference':lines+=[block['text']+f' [Source]({block["url"]}).','']
   elif kind=='table':
    lines+=[block['caption'],'','| '+' | '.join(block['headers'])+' |','| '+' | '.join(['---']*len(block['headers']))+' |']
    lines+=['| '+' | '.join(row)+' |' for row in block['rows']];lines+=['']
 (ROOT/'report'/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def write_word(pages,cfg):
 from docx import Document
 from docx.shared import Inches,Pt,RGBColor
 from docx.enum.text import WD_ALIGN_PARAGRAPH
 from docx.enum.section import WD_SECTION_START
 from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
 from docx.oxml import OxmlElement as X
 from docx.oxml.ns import qn
 from docx.opc.constants import RELATIONSHIP_TYPE as RT
 document=Document();document.core_properties.title=TITLE;document.core_properties.author='';document.core_properties.last_modified_by=''
 document.core_properties.subject='CIFAR-10 noisy-label experiment with WE and AUM'
 document.core_properties.created=datetime.now(timezone.utc);document.core_properties.modified=datetime.now(timezone.utc)
 for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
  style=document.styles[name];style.font.name='Arial';style.font.color.rgb=RGBColor(0,0,0);style.font.size=Pt(10.5)
  style.paragraph_format.space_after=Pt(4);style.paragraph_format.line_spacing=1.06
 document.styles['Title'].font.size=Pt(28);document.styles['Title'].font.bold=True
 document.styles['Subtitle'].font.size=Pt(13);document.styles['Subtitle'].font.italic=False
 for border in list(document.styles.element.xpath('.//w:pBdr')):border.getparent().remove(border)
 document.styles['Heading 1'].font.size=Pt(12);document.styles['Heading 1'].font.bold=True
 document.styles['Heading 1'].paragraph_format.space_before=Pt(9)
 document.styles['Heading 1'].paragraph_format.space_after=Pt(4)
 document.styles['Heading 1'].paragraph_format.keep_with_next=True
 document.styles['Caption'].font.size=Pt(8.5);document.styles['Caption'].font.italic=True
 section=document.sections[0];section.page_width=Inches(8.2677);section.page_height=Inches(11.6929)
 section.left_margin=section.right_margin=Inches(.72);section.top_margin=Inches(1.65);section.bottom_margin=Inches(.65)
 def cover(text,style=None,before=0,size=None):
  paragraph=document.add_paragraph(text,style);paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER
  paragraph.paragraph_format.space_before=Pt(before);paragraph.paragraph_format.space_after=Pt(0)
  paragraph.paragraph_format.line_spacing=1.0
  if size:
   for run in paragraph.runs:run.font.size=Pt(size)
  return paragraph
 cover(TITLE,'Title');cover(SUBTITLE,'Subtitle',before=18)
 cover('Project implementation and experimental evaluation',before=44,size=12)
 for i,name in enumerate(COVER_NAMES):cover(name,before=32 if i==0 else 0,size=12)
 cover('BITS Pilani, K. K. Birla Goa Campus',before=280,size=11.5)
 cover('BITS F471 - Introduction to Large Language Models',before=6,size=11.5)
 cover('Group 2',before=6,size=11.5);cover('November 2026',before=28,size=11.5)
 section=document.add_section(WD_SECTION_START.NEW_PAGE);section.top_margin=Inches(.65);section.bottom_margin=Inches(.65)
 section.footer_distance=Inches(.25);section.footer.is_linked_to_previous=False
 start=X('w:pgNumType');start.set(qn('w:start'),'1');section._sectPr.append(start)
 footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
 field=X('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
 def paragraph(text,style=None):
  value=document.add_paragraph(text,style);value.paragraph_format.widow_control=True;return value
 def hyperlink(paragraph,label,url):
  link=X('w:hyperlink');link.set(qn('r:id'),paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True))
  run=X('w:r');props=X('w:rPr');color=X('w:color');color.set(qn('w:val'),'000000');props.append(color)
  underline=X('w:u');underline.set(qn('w:val'),'single');props.append(underline);run.append(props)
  size=X('w:sz');size.set(qn('w:val'),'17');props.append(size)
  text=X('w:t');text.text=label;run.append(text);link.append(run);paragraph._p.append(link)
 for page_index,page in enumerate(pages):
  if page_index:document.add_page_break()
  for block in page:
   kind=block['kind']
   if kind=='heading':paragraph(block['text'],'Heading 1')
   elif kind=='paragraph':paragraph(block['text'])
   elif kind=='equation':
    value=paragraph('');value.alignment=WD_ALIGN_PARAGRAPH.CENTER;value.paragraph_format.space_after=Pt(1)
    value._p.append(math_tree(block['name'],cfg))
   elif kind=='reference':
    value=paragraph(block['text']+' ');value.paragraph_format.space_after=Pt(3)
    for run in value.runs:run.font.size=Pt(8.5)
    hyperlink(value,'Source',block['url'])
   elif kind=='figure':
    value=paragraph('');value.paragraph_format.keep_with_next=True
    value.add_run().add_picture(str(ROOT/'results'/'figures'/block['name']),width=Inches(block['width']))
    value.alignment=WD_ALIGN_PARAGRAPH.CENTER;paragraph(block['caption'],'Caption')
   elif kind=='table':
    caption=paragraph(block['caption'],'Caption');caption.paragraph_format.keep_with_next=True
    t=document.add_table(rows=1,cols=len(block['headers']));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for column,width in zip(t.columns,block['widths']):column.width=Inches(width)
    for cell,text in zip(t.rows[0].cells,block['headers']):cell.text=text
    header=X('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(header)
    for row in block['rows']:
     for cell,text in zip(t.add_row().cells,row):cell.text=text
    borders=X('w:tblBorders')
    for name in ['top','left','bottom','right','insideH','insideV']:
     border=X('w:'+name);border.set(qn('w:val'),'single');border.set(qn('w:sz'),'4');border.set(qn('w:color'),'D9D9D9');borders.append(border)
    t._tbl.tblPr.append(borders)
    for i,row in enumerate(t.rows):
     for j,cell in enumerate(row.cells):
      cell.width=Inches(block['widths'][j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
      properties=cell._tc.get_or_add_tcPr();padding=X('w:tcMar')
      for name in ['top','left','bottom','right']:
       node=X('w:'+name);node.set(qn('w:w'),'65');node.set(qn('w:type'),'dxa');padding.append(node)
      properties.append(padding)
      if i==0:
       shade=X('w:shd');shade.set(qn('w:fill'),'E8EEF4');properties.append(shade)
      for value in cell.paragraphs:
       value.paragraph_format.space_after=Pt(1);value.paragraph_format.space_before=Pt(1);value.paragraph_format.line_spacing=1.0
       value.alignment=WD_ALIGN_PARAGRAPH.LEFT if len(block['headers'])==2 or j==1 else WD_ALIGN_PARAGRAPH.CENTER
       for run in value.runs:run.font.size=Pt(9);run.font.bold=i==0
     prevent=X('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
    paragraph('').paragraph_format.space_after=Pt(0)
 document.save(ROOT/'report'/'IDO_AUM_Lite_Report.docx')

def main():
 cfg,m=inputs();pages=content(cfg,m);write_markdown(pages);write_word(pages,cfg)
 metadata={'configuration_sha256':configuration_fingerprint(cfg),'experiment_code':m['scope']['code'],
   'metrics_json_sha256':digest(ROOT/'results'/'metrics.json'),
   'outputs':{name:digest(ROOT/'report'/name) for name in ['report.md','IDO_AUM_Lite_Report.docx']},
   'cover_reference':{'filename':'Group_2_Research_Ideas.pdf','sha256':'109d0f05c5bd6fa290330b1c1ccd15bc3ea026a2a8072de5b5d82f28b81b7352'},
   'cover_names':COVER_NAMES,'contribution_assignments':None,'editable_math':'native OMML','figures':4}
 (ROOT/'report'/'report_sources.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
 print('Created editable DOCX and Markdown from verified experiment outputs.')
if __name__=='__main__':main()
