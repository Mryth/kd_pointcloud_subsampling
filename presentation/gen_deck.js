const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
p.author = "Projet Point Clouds + KD";
p.title = "KD & Robustesse au sous-echantillonnage";

// ---- palette (medical teal / navy) ----
const DARK="08293A", NAVY="0B3C5D", TEAL="1C7293", CYAN="2EC4B6",
      CORAL="E76F51", INK="16323D", MUTE="6B7B84", WHITE="FFFFFF",
      TINT="EAF3F3", TINT2="F2F6F7";
const HF="Cambria", BF="Calibri";
const W=13.33, H=7.5;

// ---- helpers ----
function dotMotif(s, x, y, spread, n, color, maxR){
  for(let i=0;i<n;i++){
    const r = (0.04 + Math.random()*maxR);
    s.addShape(p.ShapeType.ellipse, {x:x+Math.random()*spread, y:y+Math.random()*spread,
      w:r, h:r, fill:{color:color, transparency:20+Math.random()*50}, line:{type:"none"}});
  }
}
function title(s, t){
  s.addText(t, {x:0.6,y:0.42,w:12.1,h:0.9,fontFace:HF,fontSize:32,bold:true,color:NAVY,isTextBox:true,margin:0});
}
function card(s,x,y,w,h,fill){
  s.addShape(p.ShapeType.roundRect,{x,y,w,h,rectRadius:0.09,fill:{color:fill},line:{type:"none"},
    shadow:{type:"outer",color:"9AAEB5",blur:7,offset:2,angle:90,opacity:0.35}});
}
function numCircle(s,x,y,d,num,color){
  s.addShape(p.ShapeType.ellipse,{x,y,w:d,h:d,fill:{color:color},line:{type:"none"}});
  s.addText(String(num),{x,y,w:d,h:d,align:"center",valign:"middle",fontFace:HF,fontSize:18,bold:true,color:WHITE,isTextBox:true,margin:0});
}

// ============== S1 TITLE ==============
let s = p.addSlide(); s.background={color:DARK};
dotMotif(s, 9.7, 0.4, 3.2, 26, CYAN, 0.26);
s.addText("Projet complet — résultats finaux",{x:0.7,y:1.35,w:8,h:0.4,fontFace:BF,fontSize:14,bold:true,color:CYAN,
  isTextBox:true,margin:0,charSpacing:2});
s.addText("Distillation de connaissances &\nrobustesse au sous-échantillonnage",
  {x:0.7,y:1.85,w:11.4,h:2.0,fontFace:HF,fontSize:40,bold:true,color:WHITE,isTextBox:true,margin:0,lineSpacingMultiple:1.02});
s.addText("Nuages de points appliqués à l'imagerie médicale — classification de nodules (NoduleMNIST3D)",
  {x:0.7,y:4.0,w:11.0,h:0.7,fontFace:BF,fontSize:18,color:"CFE2E6",isTextBox:true,margin:0});
s.addText("PointNet (teacher / student)  ·  Knowledge Distillation  ·  5 seeds · tests appariés",
  {x:0.7,y:5.1,w:11,h:0.5,fontFace:BF,fontSize:14,color:"9FC2C8",isTextBox:true,margin:0});
s.addNotes("Projet autour des nuages de points en imagerie médicale, focus distillation de connaissances. Sujet imposé: robustesse au sous-échantillonnage pour la classification. Ce deck couvre le projet complet: données, baselines, distillation (5 variantes + teacher robuste), et une analyse critique qui réfute H1.");

// ============== S2 RQ & Hypothesis ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Question de recherche");
card(s,0.6,1.6,12.1,1.85,TINT);
s.addText("Q",{x:0.95,y:1.95,w:1.0,h:1.0,align:"center",valign:"middle",fontFace:HF,fontSize:30,bold:true,color:WHITE,
  fill:{color:TEAL},isTextBox:true,margin:0,shape:p.ShapeType.ellipse});
s.addText([{text:"La distillation d'un teacher entraîné sur des nuages ",options:{}},
  {text:"denses",options:{bold:true,color:TEAL}},
  {text:" permet-elle à un student léger d'être plus ",options:{}},
  {text:"robuste au sous-échantillonnage",options:{bold:true,color:TEAL}},
  {text:" des points à l'inférence, pour la classification de nodules pulmonaires ?",options:{}}],
  {x:2.25,y:1.72,w:10.1,h:1.6,fontFace:BF,fontSize:20,color:INK,valign:"middle",isTextBox:true,margin:0,lineSpacingMultiple:1.08});
card(s,0.6,3.75,12.1,1.7,TINT2);
s.addText("H1",{x:0.95,y:4.05,w:1.0,h:1.0,align:"center",valign:"middle",fontFace:HF,fontSize:26,bold:true,color:WHITE,
  fill:{color:CORAL},isTextBox:true,margin:0,shape:p.ShapeType.ellipse});
s.addText([{text:"Hypothèse.  ",options:{bold:true,color:CORAL}},
  {text:"Un student ",options:{}},{text:"distillé",options:{bold:true}},
  {text:" se dégrade plus gracieusement quand on retire des points en test qu'un student entraîné ",options:{}},
  {text:"from scratch",options:{italic:true}},{text:" (cross-entropy seule), à capacité égale.",options:{}}],
  {x:2.25,y:3.87,w:10.1,h:1.46,fontFace:BF,fontSize:18,color:INK,valign:"middle",isTextBox:true,margin:0,lineSpacingMultiple:1.08});
s.addText("Condition imposée : données médicales · dataset & architectures libres (à justifier).",
  {x:0.6,y:5.7,w:12.1,h:0.4,fontFace:BF,fontSize:13,italic:true,color:MUTE,isTextBox:true,margin:0});
s.addNotes("La question porte sur un bénéfice précis de la KD: non pas la précision brute, mais la robustesse quand le nuage de points devient parcimonieux. H1 compare student distillé vs student from scratch.");

// ============== S3 Dataset ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Données — NoduleMNIST3D (MedMNIST v2)");
const stats=[["1 633","volumes CT (1158/165/310)"],["28³","voxels par volume"],["~25 %","nodules malins (déséquilibré)"],["binaire","bénin vs malin"]];
let sx=0.6; const sw=2.95, gap=0.18;
stats.forEach((st,i)=>{ const x=sx+i*(sw+gap); card(s,x,1.65,sw,1.5,i%2?TINT2:TINT);
  s.addText(st[0],{x:x,y:1.8,w:sw,h:0.8,align:"center",fontFace:HF,fontSize:34,bold:true,color:i%2?CORAL:TEAL,isTextBox:true,margin:0});
  s.addText(st[1],{x:x+0.15,y:2.6,w:sw-0.3,h:0.5,align:"center",fontFace:BF,fontSize:12.5,color:MUTE,isTextBox:true,margin:0});});
s.addText("Pourquoi ce jeu de données",{x:0.6,y:3.5,w:11,h:0.4,fontFace:HF,fontSize:18,bold:true,color:NAVY,isTextBox:true,margin:0});
const why=["Médical & public (condition imposée), licence ouverte",
  "Léger → budget laptop-GPU : on peut entraîner teacher + student + variantes KD",
  "Convertible en nuages de points : échantillonnage de voxels pondéré par l'intensité → (x, y, z, intensité)",
  "Idéal pour le sous-échantillonnage : nuage dense 1024 pts, puis on retire des points au test"];
s.addText(why.map((t,i)=>({text:t,options:{bullet:{code:"2022",indent:18},breakLine:true,paraSpaceAfter:7}})),
  {x:0.7,y:4.0,w:12.0,h:2.6,fontFace:BF,fontSize:15.5,color:INK,isTextBox:true,margin:0});
s.addNotes("NoduleMNIST3D: petit, public, médical. On le transforme en nuages de points. Déséquilibre ~1:4 → on rapporte AUC et balanced-accuracy, pas l'accuracy brute, et on pondère la loss.");

// ============== S4 Volumes -> clouds ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Des volumes CT aux nuages de points");
const steps=[["Volume 28³","intensités CT (0–255)"],["Échantillonnage","voxels pondérés par l'intensité"],
  ["Nuage dense","1024 pts · (x, y, z, I)"],["Sous-échantillonnage","on garde N points au test"]];
const cw=2.9, cgap=0.25; let cx0=0.6;
steps.forEach((st,i)=>{ const x=cx0+i*(cw+cgap); card(s,x,1.9,cw,2.5,TINT2);
  numCircle(s,x+cw/2-0.33,2.15,0.66,i+1,i<3?TEAL:CORAL);
  s.addText(st[0],{x:x+0.1,y:3.0,w:cw-0.2,h:0.5,align:"center",fontFace:HF,fontSize:17,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText(st[1],{x:x+0.15,y:3.5,w:cw-0.3,h:0.8,align:"center",fontFace:BF,fontSize:13,color:MUTE,isTextBox:true,margin:0});
  if(i<3) s.addText("→",{x:x+cw-0.02,y:2.9,w:0.3,h:0.6,align:"center",valign:"middle",fontFace:BF,fontSize:26,bold:true,color:CYAN,isTextBox:true,margin:0});});
dotMotif(s,5.6,4.85,2.0,24,TEAL,0.14);
s.addText("Chaque nodule devient un nuage de points : coordonnées centrées et mises à l'échelle (boule unité), intensité en 4ᵉ canal.",
  {x:0.6,y:5.9,w:12.1,h:0.6,align:"center",fontFace:BF,fontSize:14,italic:true,color:MUTE,isTextBox:true,margin:0});
s.addNotes("Pipeline de conversion. On échantillonne les voxels pondérés par l'intensité pour concentrer les points sur la structure du nodule. Le protocole de robustesse retire ensuite des points d'un nuage dense fixe.");

// ============== S5 Architectures ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Architectures — PointNet (teacher / student)");
function archCard(x,color,name,params,emb,role){
  card(s,x,1.7,5.9,2.9,TINT);
  s.addShape(p.ShapeType.roundRect,{x:x,y:1.7,w:5.9,h:0.75,rectRadius:0.09,fill:{color:color},line:{type:"none"}});
  s.addText(name,{x:x+0.3,y:1.7,w:5.3,h:0.75,valign:"middle",fontFace:HF,fontSize:20,bold:true,color:WHITE,isTextBox:true,margin:0});
  s.addText(params,{x:x+0.3,y:2.65,w:5.3,h:0.9,fontFace:HF,fontSize:40,bold:true,color:color,isTextBox:true,margin:0});
  s.addText(role,{x:x+0.3,y:3.6,w:5.3,h:0.9,fontFace:BF,fontSize:14,color:INK,isTextBox:true,margin:0,lineSpacingMultiple:1.05});
}
archCard(0.6,TEAL,"Teacher","0.8 M params","1024","emb 1024 · haute capacité ·\nentraîné sur nuages denses");
archCard(6.83,CORAL,"Student","54 k params","256","emb 256 · ~15× plus léger ·\nle modèle qu'on veut rendre robuste");
s.addText([{text:"Pourquoi PointNet : ",options:{bold:true,color:NAVY}},
  {text:"classifieur canonique, invariant par permutation ; son max-pooling global unique rend la robustesse au sous-échantillonnage facile à analyser ; assez léger pour teacher + student + variantes KD. (DGCNN / Point Transformer = extensions plus lourdes possibles.)",options:{}}],
  {x:0.6,y:4.85,w:12.1,h:1.6,fontFace:BF,fontSize:15,color:INK,isTextBox:true,margin:0,lineSpacingMultiple:1.08});
s.addNotes("PointNet pour teacher et student, largeur scalable. Le choix est justifié par la simplicité, l'invariance par permutation, et surtout le max-pool global qui rend l'analyse de robustesse limpide.");

// ============== S6 Protocol ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Protocole — robustesse au sous-échantillonnage");
const prot=["Test : garder N points aléatoires, N ∈ {1024, 512, … , 16} ; AUC moyennée sur 5 tirages",
  "Chaque configuration entraînée sur 5 seeds → moyenne ± écart-type, IC 95 % bootstrap",
  "Comparaisons appariées (même seed = même teacher) : Δ AUC, t-test, seeds gagnées",
  "(T, α) choisis sur la validation uniquement ; même règle de sélection pour toutes les variantes"];
prot.forEach((t,i)=>{ const y=1.7+i*0.92; numCircle(s,0.7,y,0.6,i+1,NAVY);
  s.addText(t,{x:1.5,y:y,w:7.4,h:0.6,valign:"middle",fontFace:BF,fontSize:15,color:INK,isTextBox:true,margin:0});});
const vx=[9.4,10.7,12.0], vn=[40,16,5], vlab=["1024","128","16"];
vx.forEach((x,i)=>{ card(s,x-0.15,2.2,1.15,1.15,TINT2); dotMotif(s,x-0.05,2.3,0.9,vn[i],TEAL,0.09);
  s.addText(vlab[i]+" pts",{x:x-0.25,y:3.42,w:1.35,h:0.35,align:"center",fontFace:BF,fontSize:12,bold:true,color:MUTE,isTextBox:true,margin:0});});
s.addText("← de plus en plus parcimonieux",{x:9.2,y:3.9,w:3.4,h:0.4,align:"center",fontFace:BF,fontSize:12,italic:true,color:CORAL,isTextBox:true,margin:0});
s.addText("Pourquoi : sur 310 nodules de test, l'écart-type entre seeds (≈ 0.02 AUC) est du même ordre que les effets recherchés.",
  {x:0.6,y:5.6,w:12.1,h:0.6,fontFace:BF,fontSize:13,italic:true,color:MUTE,isTextBox:true,margin:0});
s.addNotes("Le protocole de la première moitié a été renforcé: une seule seed ne permettait pas de conclure. 5 seeds, 5 tirages de points par densité, intervalles de confiance et tests appariés. Les hyper-paramètres de KD sont choisis sur la validation, jamais sur le test.");

const cats=["1024","512","256","128","64","32","16"];
const chartOpts=(minV,maxV)=>({x:0.6,y:1.55,w:8.1,h:5.3,showTitle:false,showLegend:true,legendPos:"b",legendFontFace:BF,legendFontSize:11,
   lineSize:3,lineDataSymbol:"circle",lineDataSymbolSize:7,
   valAxisMinVal:minV,valAxisMaxVal:maxV,valAxisMajorUnit:0.05,
   catAxisTitle:"# points à l'inférence (→ plus parcimonieux)",showCatAxisTitle:true,catAxisTitleColor:MUTE,catAxisTitleFontSize:11,catAxisTitleFontFace:BF,
   valAxisTitle:"ROC-AUC (moyenne 5 seeds)",showValAxisTitle:true,valAxisTitleColor:MUTE,valAxisTitleFontSize:11,valAxisTitleFontFace:BF,
   catAxisLabelColor:MUTE,valAxisLabelColor:MUTE,catAxisLabelFontSize:11,valAxisLabelFontSize:11,
   catAxisLabelFontFace:BF,valAxisLabelFontFace:BF,
   valGridLine:{color:"E6ECEF",size:1},catGridLine:{style:"none"}});
function insightCard(head,items){
  card(s,8.95,1.7,3.75,5.0,DARK);
  s.addText(head,{x:9.2,y:1.95,w:3.3,h:0.4,fontFace:HF,fontSize:17,bold:true,color:CYAN,isTextBox:true,margin:0});
  s.addText(items.map(t=>({text:t,options:{bullet:{code:"2022",indent:16},breakLine:true,paraSpaceAfter:11,color:"E6F2F3"}})),
    {x:9.3,y:2.5,w:3.25,h:4.0,fontFace:BF,fontSize:13,isTextBox:true,margin:0,lineSpacingMultiple:1.05});
}
const R={teacher:[0.765,0.762,0.748,0.720,0.668,0.671,0.605], teacher_aug:[0.783,0.788,0.793,0.774,0.728,0.709,0.646],
  scratch:[0.763,0.758,0.746,0.714,0.660,0.654,0.585], aug:[0.791,0.796,0.798,0.779,0.743,0.726,0.648],
  kd:[0.779,0.779,0.765,0.739,0.673,0.666,0.582], kd_robust:[0.793,0.795,0.795,0.778,0.740,0.726,0.643],
  kd_robust_rt:[0.794,0.794,0.793,0.775,0.739,0.731,0.652]};

// ============== S7 Baselines (5 seeds) ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Baselines sur 5 seeds — ce qui change");
s.addChart(p.ChartType.line,[
  {name:"Teacher (0.8M)",labels:cats,values:R.teacher},
  {name:"Student scratch (54k)",labels:cats,values:R.scratch}],
  {...chartOpts(0.55,0.85),chartColors:[TEAL,CORAL]});
insightCard("À retenir",["1ʳᵉ moitié (1 seed) : student 0.810 vs teacher 0.791. Sur 5 seeds : 0.763 vs 0.765 → égalité.",
  "Le student est ~3× plus instable (σ 0.028 vs 0.008).",
  "L'avantage du teacher n'existe qu'à ≤ 32 pts (16 pts : 0.605 vs 0.585) — un signal faible à transférer."]);
s.addNotes("Les résultats single-seed de la première moitié étaient trompeurs. Sur 5 seeds teacher et student sont à égalité à pleine densité, et le teacher n'est un peu plus robuste qu'en sparsité extrême.");

// ============== S8 Variants ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Distillation — 6 variantes, une seule boucle");
const kd=[["scratch","CE seule, nuages denses"],
  ["aug — le témoin","CE seule, student voit un nuage sous-échantillonné (N ~ log-U[16, 1024]) — sans teacher"],
  ["KD logit (Hinton)","α·CE + (1-α)·T²·KL(z_s/T ‖ z_t/T), teacher gelé, entrées denses"],
  ["KD robuste","teacher voit le nuage dense, student un nuage sous-échantillonné"],
  ["+ KD de features","+ alignement des vecteurs globaux g (projection 256 → 1024)"],
  ["KD robuste, teacher robuste","teacher lui-même entraîné avec sous-échantillonnage"]];
kd.forEach((k,i)=>{ const x=0.6+(i%3)*4.1, y=1.7+Math.floor(i/3)*2.35; card(s,x,y,3.9,2.1,i==1?TINT2:TINT);
  numCircle(s,x+0.25,y+0.28,0.58,i+1,i==1?CORAL:TEAL);
  s.addText(k[0],{x:x+1.0,y:y+0.27,w:2.8,h:0.6,valign:"middle",fontFace:HF,fontSize:15.5,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText(k[1],{x:x+0.3,y:y+1.0,w:3.4,h:1.0,fontFace:BF,fontSize:12.5,color:INK,isTextBox:true,margin:0,lineSpacingMultiple:1.05});});
s.addText("Le témoin « aug » voit exactement les mêmes entrées que la KD robuste : tout écart entre les deux est attribuable au teacher.",
  {x:0.6,y:6.5,w:12.1,h:0.5,fontFace:BF,fontSize:13,italic:true,color:MUTE,isTextBox:true,margin:0});
s.addNotes("Toutes les variantes partagent le même code (src/kd.py). Le point méthodologique central: le témoin aug. Sans lui, impossible de séparer l'effet de la distillation de l'effet de l'augmentation par sous-échantillonnage. T=2, alpha=0.7 choisis sur la validation.");

// ============== S9 Results chart ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Résultats — courbes de robustesse (AUC test)");
s.addChart(p.ChartType.line,[
  {name:"Teacher dense",labels:cats,values:R.teacher},
  {name:"Scratch",labels:cats,values:R.scratch},
  {name:"KD logit",labels:cats,values:R.kd},
  {name:"Aug (témoin)",labels:cats,values:R.aug},
  {name:"KD robuste",labels:cats,values:R.kd_robust}],
  {...chartOpts(0.55,0.85),chartColors:[NAVY,"A9B6BC","8FB8C4",CORAL,CYAN]});
insightCard("À retenir",["KD robuste ≫ scratch : +0.03 → +0.08 AUC, 5/5 seeds, p ≤ 0.01.",
  "Mais aug (sans teacher) fait la même chose : les deux courbes se superposent.",
  "KD logit classique : gain faible et non significatif (p ≥ 0.08)."]);
s.addNotes("Deux groupes de courbes: les modèles entraînés sur nuages denses en bas, ceux entraînés avec sous-échantillonnage en haut. La KD robuste et le témoin aug sont indiscernables.");

// ============== S10 Decisive test ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Le test décisif — KD robuste vs témoin aug");
const st=[["+0.08","KD robuste − scratch\n(64 pts, p < 0.01, 5/5)",TEAL],["+0.08","aug − scratch\n(64 pts, p < 0.01, 5/5)",TEAL],
  ["≈ 0","KD robuste − aug\n(|Δ| ≤ 0.004, p ≥ 0.69)",CORAL],["≈ 0","+ features − KD robuste\n(non significatif)",CORAL]];
st.forEach((v,i)=>{ const x=0.6+i*(2.95+0.18); card(s,x,1.65,2.95,1.9,i%2?TINT2:TINT);
  s.addText(v[0],{x:x,y:1.8,w:2.95,h:0.8,align:"center",fontFace:HF,fontSize:34,bold:true,color:v[2],isTextBox:true,margin:0});
  s.addText(v[1],{x:x+0.15,y:2.6,w:2.65,h:0.85,align:"center",fontFace:BF,fontSize:12.5,color:MUTE,isTextBox:true,margin:0});});
s.addText("Lecture",{x:0.6,y:3.95,w:11,h:0.4,fontFace:HF,fontSize:18,bold:true,color:NAVY,isTextBox:true,margin:0});
const lec=["Lu seul, « KD robuste vs scratch » ressemble à un succès net de la distillation",
  "Le témoin montre que tout le gain vient de l'entraînement sur nuages parcimonieux, pas des cibles douces du teacher",
  "Sans ce témoin, on aurait publié un faux positif"];
s.addText(lec.map(t=>({text:t,options:{bullet:{code:"2022",indent:18},breakLine:true,paraSpaceAfter:7}})),
  {x:0.7,y:4.45,w:12.0,h:2.2,fontFace:BF,fontSize:15.5,color:INK,isTextBox:true,margin:0});
s.addNotes("C'est le résultat central. La différence appariée KD robuste moins aug est nulle à toutes les densités. La robustesse vient de l'augmentation.");

// ============== S11 Robust teacher ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Et avec un teacher robuste ?");
s.addChart(p.ChartType.line,[
  {name:"Teacher robuste (0.8M)",labels:cats,values:R.teacher_aug},
  {name:"Aug (student 54k)",labels:cats,values:R.aug},
  {name:"KD depuis teacher robuste",labels:cats,values:R.kd_robust_rt}],
  {...chartOpts(0.6,0.85),chartColors:[NAVY,CORAL,CYAN]});
insightCard("À retenir",["Le teacher entraîné avec sous-échantillonnage reste sous le student aug (0/5 seeds gagnées à 64 pts).",
  "KD depuis lui − aug ≈ 0 (p ≥ 0.45).",
  "Plus de poids au teacher (α = 0.3) dégrade le student (−0.022 à 16 pts, p = 0.02)."]);
s.addNotes("Hypothèse testée: le teacher dense n'avait pas de robustesse à transmettre. On l'a donc rendu robuste. Résultat: il reste moins bon que le petit modèle augmenté — 1158 nodules ne suffisent pas à exploiter 0.8M paramètres. La KD ne peut transférer que ce que le teacher sait de plus.");

// ============== S12 Critical analysis ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Analyse critique");
const find=[["H1 est réfutée","Aucune variante de KD (logits, features, teacher dense ou robuste, deux α) ne bat le simple sous-échantillonnage à l'entraînement.",CORAL],
  ["Pourquoi : le teacher n'en sait jamais plus","Petit dataset (1158 nodules) → la capacité du teacher sur-apprend ; il n'a rien à transférer au-delà de ce que le student apprend seul.",TEAL],
  ["Le seuil dépend de la densité","L'AUC survit, mais la balanced accuracy tombe à 0.5 en basse densité (tout « bénin ») pour les modèles denses, et inversement → calibrer le seuil par N.",NAVY]];
find.forEach((f,i)=>{ const y=1.65+i*1.62; card(s,0.6,y,12.1,1.45,i==1?TINT2:TINT);
  s.addText((i+1),{x:0.95,y:y+0.28,w:0.9,h:0.9,align:"center",valign:"middle",fontFace:HF,fontSize:30,bold:true,color:f[2],isTextBox:true,margin:0});
  s.addText(f[0],{x:2.0,y:y+0.17,w:10.4,h:0.5,fontFace:HF,fontSize:18,bold:true,color:INK,isTextBox:true,margin:0});
  s.addText(f[1],{x:2.0,y:y+0.65,w:10.5,h:0.72,fontFace:BF,fontSize:14,color:"33474F",isTextBox:true,margin:0,lineSpacingMultiple:1.03});});
s.addNotes("Résultat négatif propre et défendable. Bonus: le sous-échantillonnage régularise aussi à pleine densité (+0.03 AUC). Et un problème pratique mis en évidence: le seuil de décision n'est valable que pour une densité.");

// ============== S13 Limits & perspectives ==============
s = p.addSlide(); s.background={color:WHITE}; title(s,"Limites & perspectives");
const lp=[["Taille du nodule effacée","la normalisation à la boule unité supprime la taille (indice clé de malignité) → AUC plafonnée ~0.8"],
  ["Puissance statistique","310 nodules test, 5 seeds : un effet KD de ±0.01 resterait indétectable"],
  ["Un teacher vraiment supérieur","CNN 3D sur voxels distillé vers le student nuage de points (KD cross-modale)"],
  ["Seuil & sous-échantillonnage","seuil calibré par densité ; sous-échantillonnage structuré (trous locaux, coupes)"]];
lp.forEach((k,i)=>{ const x=0.6+(i%2)*6.2, y=1.7+Math.floor(i/2)*2.35; card(s,x,y,5.9,2.05,i<2?TINT2:TINT);
  numCircle(s,x+0.3,y+0.3,0.62,i+1,i<2?CORAL:TEAL);
  s.addText(k[0],{x:x+1.1,y:y+0.32,w:4.6,h:0.6,valign:"middle",fontFace:HF,fontSize:16.5,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText(k[1],{x:x+0.35,y:y+1.05,w:5.2,h:0.9,fontFace:BF,fontSize:13.5,color:INK,isTextBox:true,margin:0,lineSpacingMultiple:1.05});});
s.addNotes("Limites en corail, pistes en teal. La piste la plus prometteuse: une KD cross-modale depuis un modèle voxel qui a accès à de l'information que le nuage de points n'a pas.");

// ============== S14 Conclusion ==============
s = p.addSlide(); s.background={color:DARK};
dotMotif(s,0.3,4.3,3.0,22,TEAL,0.22);
s.addText("Conclusion",{x:0.7,y:0.9,w:11,h:0.9,fontFace:HF,fontSize:34,bold:true,color:WHITE,isTextBox:true,margin:0});
const concl=["Le sous-échantillonnage dégrade nettement la classification (AUC ~0.77 → ~0.59 pour les modèles denses).",
  "Entraîner sur des nuages parcimonieux le corrige : +0.03 → +0.08 AUC, à toutes les densités.",
  "La distillation n'ajoute rien au-delà de cette augmentation : H1 réfutée, avec le bon témoin.",
  "Leçon : une KD n'aide que si le teacher sait quelque chose que le student ne peut pas apprendre seul."];
s.addText(concl.map(t=>({text:t,options:{bullet:{code:"2022",indent:20},breakLine:true,paraSpaceAfter:13,color:"E6F2F3"}})),
  {x:0.75,y:2.1,w:11.7,h:3.8,fontFace:BF,fontSize:17,isTextBox:true,margin:0,lineSpacingMultiple:1.06});
s.addText("Code reproductible : src/ · prepare_data.py · run_seeds.py · run_kd.py · run_robust_teacher.py · notebook",
  {x:0.75,y:6.55,w:12,h:0.5,fontFace:BF,fontSize:13,color:"9FC2C8",isTextBox:true,margin:0});
s.addNotes("Résumé: résultat négatif mais rigoureux. La méthodologie (5 seeds, tests appariés, témoin aug) est ce qui permet de l'affirmer.");

p.writeFile({fileName:"KD_pointcloud_presentation.pptx"}).then(f=>console.log("wrote",f));
