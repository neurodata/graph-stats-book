Search.setIndex({docnames:["applications/ch10/anomaly-detection","applications/ch10/ch10","applications/ch10/significant-communities","applications/ch10/significant-edges","applications/ch10/significant-vertices","applications/ch8/anomaly-detection","applications/ch8/ch8","applications/ch8/community-detection","applications/ch8/model-selection","applications/ch8/out-of-sample","applications/ch8/testing-differences","applications/ch8/vertex-nomination","applications/ch9/ch9","applications/ch9/graph-matching-vertex","applications/ch9/two-sample-hypothesis","applications/ch9/vertex-nomination","foundations/ch1/ch1","foundations/ch1/examples-of-applications","foundations/ch1/exercises","foundations/ch1/types-of-learning-probs","foundations/ch1/types-of-networks","foundations/ch1/what-is-a-network","foundations/ch1/why-study-networks","foundations/ch2/ch2","foundations/ch2/discover-and-visualize","foundations/ch2/fine-tune","foundations/ch2/get-the-data","foundations/ch2/prepare-the-data","foundations/ch2/select-and-train","foundations/ch2/transformation-techniques","foundations/ch3/big-picture","foundations/ch3/ch3","foundations/ch3/discover-and-visualize","foundations/ch3/get-the-data","foundations/ch3/prepare-the-data","intro","representations/ch4/ch4","representations/ch4/matrix-representations","representations/ch4/network-representations","representations/ch4/properties-of-networks","representations/ch4/regularization","representations/ch5/ch5","representations/ch5/models-with-covariates","representations/ch5/multi-network-models","representations/ch5/single-network-models","representations/ch5/why-use-models","representations/ch6/ch6","representations/ch6/estimating-parameters","representations/ch6/graph-neural-networks","representations/ch6/joint-representation-learning","representations/ch6/multigraph-representation-learning","representations/ch6/random-walk-diffusion-methods","representations/ch6/why-embed-networks","representations/ch7/ch7","representations/ch7/theory-matching","representations/ch7/theory-multigraph","representations/ch7/theory-single-network","representations/single-network-models"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["applications/ch10/anomaly-detection.ipynb","applications/ch10/ch10.ipynb","applications/ch10/significant-communities.ipynb","applications/ch10/significant-edges.ipynb","applications/ch10/significant-vertices.ipynb","applications/ch8/anomaly-detection.ipynb","applications/ch8/ch8.ipynb","applications/ch8/community-detection.ipynb","applications/ch8/model-selection.ipynb","applications/ch8/out-of-sample.ipynb","applications/ch8/testing-differences.ipynb","applications/ch8/vertex-nomination.ipynb","applications/ch9/ch9.ipynb","applications/ch9/graph-matching-vertex.ipynb","applications/ch9/two-sample-hypothesis.ipynb","applications/ch9/vertex-nomination.ipynb","foundations/ch1/ch1.ipynb","foundations/ch1/examples-of-applications.ipynb","foundations/ch1/exercises.ipynb","foundations/ch1/types-of-learning-probs.ipynb","foundations/ch1/types-of-networks.ipynb","foundations/ch1/what-is-a-network.ipynb","foundations/ch1/why-study-networks.ipynb","foundations/ch2/ch2.ipynb","foundations/ch2/discover-and-visualize.ipynb","foundations/ch2/fine-tune.ipynb","foundations/ch2/get-the-data.ipynb","foundations/ch2/prepare-the-data.ipynb","foundations/ch2/select-and-train.ipynb","foundations/ch2/transformation-techniques.ipynb","foundations/ch3/big-picture.ipynb","foundations/ch3/ch3.ipynb","foundations/ch3/discover-and-visualize.ipynb","foundations/ch3/get-the-data.ipynb","foundations/ch3/prepare-the-data.ipynb","intro.md","representations/ch4/ch4.ipynb","representations/ch4/matrix-representations.ipynb","representations/ch4/network-representations.ipynb","representations/ch4/properties-of-networks.ipynb","representations/ch4/regularization.ipynb","representations/ch5/ch5.ipynb","representations/ch5/models-with-covariates.ipynb","representations/ch5/multi-network-models.ipynb","representations/ch5/single-network-models.ipynb","representations/ch5/why-use-models.ipynb","representations/ch6/ch6.ipynb","representations/ch6/estimating-parameters.ipynb","representations/ch6/graph-neural-networks.ipynb","representations/ch6/joint-representation-learning.ipynb","representations/ch6/multigraph-representation-learning.ipynb","representations/ch6/random-walk-diffusion-methods.ipynb","representations/ch6/why-embed-networks.ipynb","representations/ch7/ch7.ipynb","representations/ch7/theory-matching.ipynb","representations/ch7/theory-multigraph.ipynb","representations/ch7/theory-single-network.ipynb","representations/single-network-models.ipynb"],objects:{},objnames:{},objtypes:{},terms:{"00000795":[],"00019536957141698985":[],"00043756":[],"00090108":[],"0009010833022217993":[],"00413":49,"004155975322328694":[],"005625861749316":[],"00562586174932":[],"017044913788664812":[],"017126149668054324":[],"017149083444858545":[],"017294208713800992":[],"019168184381196746":[],"02726":49,"027389678814754834":[],"05816":49,"07680":[],"08186263117598762":[],"08318":[],"08319":[],"08336":[],"08342":[],"08600743004374037":[],"08754":49,"08840747693989694":[],"09025788904156264":[],"09136419645543331":[],"09161704113449845":[],"09337":[],"100":[41,49],"104":49,"1052":49,"1053":49,"1054":49,"1055":49,"1056":49,"1093":49,"11692":49,"12426":[],"129":49,"137":49,"14631":49,"148":49,"149":49,"150":49,"1500":49,"151":49,"152":49,"172":49,"173":49,"174":49,"175":49,"176":49,"177":49,"178":49,"179":49,"180":49,"1936":49,"1959":[44,57],"1982":49,"1ebf56eac912":[],"20105":[],"2017":49,"22945":[],"22972":49,"235931277557181e":[],"2359312775571826e":[],"24767":[],"24805":[],"24852":[],"25910":49,"26147":[],"274":49,"275":49,"276":49,"277":49,"278":49,"28080":[],"282":[],"283":[],"284":[],"285":[],"286":[],"28849":49,"290":[44,57],"297":[44,57],"298":44,"300":49,"301":49,"302":[49,57],"303":49,"304":49,"3144":[],"31787":49,"321":49,"32529":[],"361":49,"37636019d9b2":49,"377":49,"37789":[],"40128":49,"40775285877078155":[],"41862":49,"42766":49,"43034":49,"43067":49,"43171":49,"43248":49,"43305":49,"43349":49,"43381":49,"43407":49,"43426":49,"43441":49,"43454":49,"43465":49,"43475":49,"43482":49,"43490":49,"43496":49,"43502":49,"43507":49,"43512":49,"43518":49,"43520":49,"43525":49,"43528":49,"44406":[],"46005":49,"47159":[],"48944":49,"52632":[],"556":49,"557":49,"558":49,"559":49,"560":49,"57285":49,"60223":49,"60732":[],"63162":49,"66100":49,"673872836847387":[],"69900":[],"70282":[],"70563":[],"70729":[],"70832":[],"70897":[],"70932":[],"70956":[],"70969":[],"70978":[],"72484":[],"74441":49,"760":49,"761":49,"762":49,"763":49,"764":49,"765":49,"766":49,"767":49,"77380":49,"80318":49,"83256":49,"83676":[],"85159":[],"8581568507e8":[44,57],"8806372446165007":[],"88659":49,"91598":49,"931":49,"932":49,"933":49,"934":49,"935":49,"94536":49,"97474":49,"case":[41,44,57],"class":[41,44,49,57],"final":44,"float":49,"function":49,"import":[41,44,49,57],"new":[41,44,49,57],"return":49,"short":[],"true":[41,44,49,57],"try":[44,49],Being:[],But:49,For:[41,44,49,57],One:49,The:[41,44,49,57],Then:49,There:49,These:[44,49,57],Using:[44,57],__call__:49,_backend:49,_cond:49,_emb:[],_event:49,_fit_transform:49,_flag:49,_get_tuning_paramet:49,_ll:49,_output:49,_start_tim:49,_xxt:49,abl:49,about:[41,44,49,57],abov:[41,44,49,57],access:49,accomplish:49,accur:57,accuraci:41,achiev:49,acquir:49,actual:[41,44,49,57],added:49,addit:49,adj:[44,57],adjac:[44,49,57],affect:[44,57],after:49,again:41,against:[],age:[41,44,49,57],agreement:49,algebra:49,algorithm:[44,49,57],align:[44,49],all:[41,44,49,57],allow:[41,49],almost:[44,49,57],alpha:[],alpha_:49,alreadi:[44,49,57],also:[44,49],alwai:[44,57],amax:49,amin:49,amount:[41,49],analysi:49,analyt:[],ani:[41,44,57],anoth:[41,44,57],answer:[41,44,57],anticip:[],anyth:[],apart:49,aphor:[41,44,57],appar:[44,57],appli:49,applic:41,approach:[41,44,57],appropri:[41,44,57],arbitrari:[],aren:[44,49],arg:49,argument:[],arr:[44,57],arrai:[44,49,57],arrang:[44,57],ascend:49,aspect:[41,44,57],assign:[44,57],associ:49,assort:49,assum:[41,44,49,57],assumpt:41,asx008:49,attent:[41,44,57],attribut:[41,44,57],avail:[],averag:[44,57],axes:49,axessubplot:[44,57],axi:49,axs:49,axx:49,base:49,basi:[41,44,57],becaus:[41,44,49,57],becom:[44,57],befor:[41,44,49,57],begin:[44,49],behav:[44,57],behavior:[44,57],being:[41,44,57],believ:[41,44,57],belong:49,below:[44,49,57],ben:35,bern:[],bernoulli:[41,44,49,57],best:[44,49,57],best_alpha:49,beta:49,between:[41,44,49,57],bia:41,big:49,biggest:49,binari:49,binkiewicz:49,biomet:49,biometrika:49,bit:49,black:49,bmatrix:49,book:41,both:49,box:[41,44,57],brack:49,brain:49,breadth:41,british:[41,44,57],calcul:44,call:[41,44,49,57],can:[41,44,49,57],candid:[41,44,57],capabl:49,captur:41,care:[44,57],casc:49,cbar:49,cbar_kw:49,cdot:[44,49],center:[44,49,57],centuri:[41,44,57],chanc:[41,44,57],chang:49,chapter:44,character:[41,44],characterist:49,check:[44,49,57],choic:[41,44,49,57],choos:57,circl:41,clarifi:[41,44,57],classic:49,clearli:[44,49,57],close:[41,49],cluster:49,cmap:49,code:[44,49,57],coeffici:49,coin:41,collect:49,color:49,colorbar:49,column:49,combin:49,come:[41,49],common:[41,44,57],commun:[44,49,57],complet:41,complex:[41,44,57],complic:49,comput:[44,57],computation:49,concat:49,conceiv:[],concept:41,concern:[],conclud:44,connect:[41,44,49,57],connectom:35,consequ:44,consid:[41,44,57],constrained_layout:49,contain:49,context:[41,44,57],contribut:49,convei:[41,44,57],convolut:41,correct:41,correctli:[41,44,49],correl:49,correspond:[44,57],could:[41,44,57],counti:41,covariateassistedembed:49,covariateassistedspectralembed:49,cover:41,crappi:[],creat:49,credit:35,crucial:41,custom:[],dad:49,danc:41,dark:[44,57],darker:49,data:[41,44,49,57],datafram:[],dataset:49,deal:[44,57],debrecen:[44,57],decompos:49,decomposit:49,deduc:[],def:49,defin:[44,49,57],definit:[44,57],deg:44,delin:[44,57],denot:[44,57],depend:[41,44,57],deprec:[44,57],describ:[41,44,49,57],descript:41,design:[44,57],despit:41,detail:49,determin:[41,44,57],determinist:41,develop:[41,44,57],devis:[],df1:[],df2:[],diag_indices_from:49,diagon:[44,57],dict:49,did:[41,49],differ:[41,44,49,57],difficult:[44,49,57],dimens:49,dimension:49,direct:[44,49,57],directli:[41,44,49,57],discern:[44,57],discuss:44,displai:49,distanc:[],distinct:49,distinguish:49,distribut:[41,44,49,57],document:49,doe:[44,49,57],doesn:[41,44,49,57],doi:49,don:[41,49],done:49,dot:49,down:49,draw:49,drop:[],dropbox:[],due:[44,57],each:[41,44,49,57],earlier:49,easili:[44,57],edg:[41,49],effect:[41,49],effici:49,eigenvalu:49,eigenvector:[],eigvalsh:49,either:[44,57],elapsed_tim:49,elegan:35,element:49,elif:49,els:49,emb:49,embedding_alg:49,emphas:[41,49],end:[44,49,57],enough:[44,49,57],enti:[44,57],entri:[44,57],equal:49,equat:[],equival:[44,57],er_n:44,er_np:[44,57],erestim:[44,57],error:[44,57],especi:44,essenti:49,estim:[41,44,57],evalu:[],even:[41,44,49,57],ever:[44,57],everi:[41,44,49,57],everyth:41,exact:[44,57],exactli:[41,44,49,57],examin:44,exampl:[41,44,49,57],exce:[44,57],exceed:[],exercis:[],exist:[44,57],expand:44,expect:[41,44,57],expens:49,explain:[44,57],exploit:[],explor:[49,57],extend:[41,49],extmath:49,extra:49,extract:41,extrem:41,facilit:41,fact:[41,44,57],factor:[41,44,57],fairli:49,faithfulli:41,fall:[44,57],fals:[44,49,57],famili:[41,44,57],familiar:41,far:49,faster:49,featur:49,few:[41,49],fig:49,fight:41,figsiz:49,figur:[44,49],filterwarn:49,find:[44,49,57],fine:[],finit:44,first:[41,44,49,57],firstli:[],fit:[44,49,57],fit_transform:49,fix:[41,49],flip:[41,49],follow:[41,44,49,57],fomal:[44,57],font_scal:49,form:49,formal:[44,57],fortun:[44,57],frac:49,fraction:[44,57],framework:[41,44,57],friend:[41,44,57],friendship:57,from:[41,44,49,57],from_list:[],front:[],full:49,fundament:[],further:[44,57],futur:[44,57],futurewarn:[44,57],game:49,gca:49,gen_covari:49,gen_sbm:49,gender:49,gener:[41,49],geomspac:[],georg:[41,44,57],get:[41,44,57],get_eigv:49,getattr:49,give:[44,49,57],given:[41,44,57],glue:49,goal:[44,49,57],golden:49,good:[],gotit:49,govern:[44,57],grade:[41,44,57],graph:49,graspolog:[44,57],greater:49,group:[41,44,49,57],grow:[44,57],had:[41,44,57],half:44,hand:[41,44,57],happen:[44,49],has:[41,44,49,57],have:[41,44,49,57],head:41,heatmap:[44,49,57],help:[44,49],henceforth:49,here:49,high:44,higher:[41,44,49,57],highest:49,hold:[41,44],holist:[],hollow:44,hopkin:35,hostedtoolcach:49,hotel:49,how:[41,44,49,57],howev:49,html:49,http:49,hue:49,hypothet:[],idea:[44,57],ideal:[],ident:[44,57],ieee:49,ignor:49,illustr:[44,49,57],imagin:41,impact:[41,44,57],imperfect:49,implement:49,importlib:49,imposs:[44,49],improv:49,incid:[44,57],includ:[44,57],incorpor:41,increas:[44,57],inde:[44,57],index:[41,44,57],indic:[41,44,57],indistinguish:49,inertia:49,inertia_:[],inertia_tri:49,inertias_:49,infer:41,inferenti:[],influenc:[44,57],inform:[41,44,49,57],initi:49,input:[44,49,57],instanc:[41,44,49,57],instanti:[44,57],instead:[41,44,49,57],interest:[41,44,57],interpret:[44,49,57],intim:41,intuit:[41,44,57],intuition:[44,57],invert:[],investig:[44,49,57],ipython:[44,49,57],isn:41,issu:49,iter:49,its:[44,49,57],itself:[],job:49,joblib:49,john:35,jointli:49,june:49,just:[41,44,49,57],keep:[],kei:[],keyboardinterrupt:49,kind:49,kmean:49,knew:[41,44,57],know:[41,44,57],knowledg:41,known:49,kwarg:49,l_ax:49,l_eigval:49,l_latent:49,l_top:49,label:49,labels_:49,lack:41,lambda:49,lambda_1:49,lambda_:49,lambda_i:49,lambda_k:49,lambda_r:49,land:41,laplacian:49,laplacianspectralemb:49,larg:[41,49],larger:[44,49,57],last:49,latent:49,latent_left_:[],latent_posit:49,latent_right_:49,latents_:49,later:[41,44,49,57],layer:49,lead:49,learn:[41,44,57],least:49,left:[44,49],len:[],length:49,less:49,let:[41,44,49],level:[41,44,57],leverag:41,lib:49,life:41,lighter:49,like:[41,44,49,57],linalg:49,line:[44,49],linear:[44,49],linearsegmentedcolormap:[],linewidth:49,linspac:[],listedcolormap:49,littl:[44,57],lloyd:49,locat:49,longer:49,look:[41,44,49,57],loop:[44,57],loopi:44,lot:[44,57],lower:[41,44,49],lse:49,luck:49,machin:[41,49],mai:[44,57],make:[49,57],make_commun:49,manag:49,mani:[41,44,49],math:[44,49,57],mathbb:44,mathcal:[44,57],matplotlib:49,matric:[44,49,57],matrix:[44,49,57],matter:49,max:49,maxfun:[],maximum:49,mean:[41,44,57],measur:49,meet:41,member:[44,57],mere:41,merit:[],messag:49,method:49,metric:49,might:[41,44,49,57],min:49,minibatchkmean:[],minimum:49,mirror:41,miss:[44,57],model:35,modul:49,more:[41,44,49,57],most:49,mtx:[44,57],much:[41,44,49,57],multidimension:[44,57],multipli:49,multiprocess:49,must:[41,44,57],myst_nb:49,n_cluster:49,n_compon:49,n_covari:49,n_eigval:49,n_job:49,n_vertic:49,name:[],natur:49,ncol:49,ndd:[],need:[41,44,49,57],neq:[44,57],network:49,neuron:49,never:[41,44,57],new_lat:49,next:[41,44,57],nice:49,nit:49,node:[41,44,49,57],nois:49,non:[44,49,57],none:49,nor:[41,44],normal:49,note:[44,57],noth:[41,44,57],notic:49,now:49,nrow:49,num:[],number:[44,49,57],numpi:[44,49,57],obei:[],object:[],observ:[41,44,57],obviou:[],occur:[44,57],off:[44,57],often:[41,49],okai:[44,57],old:[41,44,57],one:[44,49,57],ones:49,onli:[41,44,49,57],oper:49,operand:[],opt:49,optim:49,optimizeresult:[],option:49,order:[44,49,57],org:49,organ:49,origin:49,other:[41,44,49,57],otherwis:[44,49,57],our:[41,44,57],ourselv:41,out:[44,49],outcom:41,output:49,over:44,overlai:49,overlap:49,own:[44,49,57],packag:49,page:49,pai:[41,44,57],pair:[41,44,57],pairplot:49,palett:49,panda:49,paper:49,parallel:49,paramet:[41,44,49,57],parlanc:49,particular:[41,44,49,57],particularli:[],pattern:[44,57],pcm:49,pedigo:35,peopl:[41,44,57],per:49,perf_count:[],perfect:41,perfectli:49,perhap:41,permut:[44,57],person:[41,49],phd:35,physic:49,pick:49,piec:[],pioneer:[41,44,57],place:[41,49],plai:49,plot:[44,49,57],plot_lat:49,plotting_context:49,plt:49,pmb:[41,44,57],point:[41,44],pool:49,popular:44,posit:[49,57],possess:[41,44,57],possibl:[41,44,49,57],potenti:[],practic:[41,44,57],preced:[44,57],precis:41,predetermin:49,prefer:[41,49],preprocess:49,present:49,pretti:[],previous:49,primari:[41,44,49,57],principl:[],print:[44,57],prior:49,priorit:41,probabl:[41,44,49,57],problem:49,procedur:[49,57],process:[41,44,49,57],produc:[41,44,49,57],product:49,promis:49,properti:[44,57],propos:[44,57],provid:[44,57],publ:[44,57],purpos:57,put:[],pyplot:49,python3:49,python:[44,49,57],quantiti:[41,44],quantiz:49,question:[41,44,57],quicker:[],quickli:49,quit:[44,57],rais:49,ram:[44,57],random:49,randomized_svd:49,randomli:[44,57],rare:41,rather:[41,44,57],ratio:49,readi:49,realist:[44,57],realiz:[41,44,57],reason:[41,44,49],recal:44,recent:49,red:[44,57],reduc:49,refer:41,refin:[41,44,57],regardless:[44,57],region:49,regular:49,rel:[44,57],relat:[41,49],reload:49,remaind:[],rememb:[41,44],reorder:[44,57],replac:[44,57],repres:49,requir:41,reset_index:[],resort:[44,57],respect:[44,57],rest:[41,44,57],restor:49,result:[44,49,57],retriev:49,retrieval_context:49,return_label:49,revers:[44,49,57],right:[44,49],rocket_r:49,rohe:49,role:[],roughli:49,row:49,rvs:49,sai:[41,44,57],said:[41,44,57],same:[41,44,49,57],sampl:[44,57],sbm:49,sbm_n:[44,57],scale:[44,49,57],scatterplot:49,scenario:41,school:[41,44,57],scienc:57,scientif:[41,44,57],scientist:[41,44,57],scikit:49,scipi:49,score:49,seaborn:49,second:[41,44,49,57],section:[41,44,49,57],section_search:49,see:[41,44,49,57],seed:49,seek:44,seem:[44,49,57],select:[41,44,57],selectsvd:49,self:49,sens:[41,44,57],separ:49,seq:[44,57],sequenc:[44,57],set1:49,set:[41,44,57],set_frame_on:49,set_tick:49,set_ticklabel:49,set_titl:49,set_vis:49,shape:49,share:[49,57],should:[44,49],show:[44,49,57],shown:[],shrink:49,signal:49,silhouett:49,silhouette_scor:49,sim:[44,57],similar:[44,49,57],similarli:49,simpl:[41,44,49,57],simpler:[41,49],simplest:[41,44,57],simpli:[44,49,57],simplic:[41,44,57],simul:[44,49,57],sinc:[44,49],singl:[41,49],singular:49,site:49,situat:[44,49,57],size:[44,49,57],sklearn:49,slightli:41,slower:[],small:[44,49,57],smaller:49,sns:49,social:[41,44,49,57],solut:[],some:[41,44,49,57],somebodi:41,somehow:49,someon:41,someth:[44,57],sometim:49,somewhat:49,somewher:49,spars:[44,57],specif:41,specifi:44,speed:49,spend:49,squar:[44,49,57],stabl:49,standard:49,start:[44,49,57],stat:49,state:[41,44,49,57],statist:[44,49,57],statistician:[41,44,57],stick:49,still:[41,44,57],stochast:41,store:49,str:49,straightforward:49,structur:[41,49],structuur:[44,57],struuctur:[44,57],stuart:49,student:[35,41,44,57],studi:44,subgraph:[44,57],subplot:49,subset_by_index:49,sum:[44,49,57],sum_:44,summar:[41,57],supports_timeout:49,suppos:[41,44,49,57],sure:49,suspici:41,svd:[],symmetr:[44,57],symmetri:[44,57],system:41,tabl:41,tail:41,take:[41,44,49,57],taken:[41,44],talk:49,task:[],tau:[44,49,57],tau_i:[44,57],techniqu:[44,49,57],tell:49,ten:[],tend:[44,57],term:[44,57],test:49,textrm:[],than:[41,44,49,57],thei:[41,44,49,57],them:49,theoret:[44,57],theori:49,therefor:[41,44,57],thi:[41,44,49,57],thing:[41,44,57],think:[41,44,57],third:49,though:[41,44,49,57],thread:49,three:[41,49],through:[49,57],thte:[],tight_layout:49,time:[41,44,49,57],timeout:49,timeouterror:49,titl:[44,49,57],to_laplacian:49,todo:49,togeth:[49,57],too:[41,44,57],tool:49,top:49,top_eigv:49,topolog:[41,44,49],toss:41,total:[44,49,57],traceback:49,tractabl:[],tradeoff:41,tradit:41,transact:49,transpos:49,trick:49,trivial:[],truncat:49,tune:49,tuning_rang:[],tuning_run:49,tupl:[44,49,57],turn:49,tutori:49,two:[41,44,49,57],type:[44,49],typeerror:[],uncertainti:41,under:57,underli:41,understand:[41,49],undirect:44,uninform:49,uniqu:[44,57],unit:49,univers:35,unless:[44,57],unlik:[44,57],unsupport:[],until:49,updat:[],upon:[44,57],use:[41,44,49,57],used:[41,44,49,57],useful:[41,44,49,57],uses:[],using:[41,44,49,57],usual:41,util:[41,49],v_i:[44,57],v_j:[44,57],valu:[41,44,49,57],valuabl:41,variabl:41,varianc:41,variant:49,vec:[44,57],vector:[44,49,57],verbos:49,veri:[41,44,49,57],version:49,vertex:[44,57],vertic:[44,57],vetex:[44,57],viewpoint:57,virtual:[44,57],visual:[44,49,57],vocabulari:41,vogelstein:49,volum:49,vstack:49,vtx_perm:[44,57],wai:[41,44,49,57],wait:49,waiter:49,want:[41,44,49,57],warn:49,weight:[44,57],well:[41,44,49,57],were:[41,44,49,57],weren:41,what:[41,44,49],when:[41,44,49,57],where:[41,44,49,57],wherein:[44,57],whether:[41,44,57],which:[41,44,49,57],white:[44,49,57],who:41,why:[44,57],wiki:49,wikipedia:49,willing:49,wish:[],wite:[44,57],within:[41,44,49,57],without:[44,57],word:[44,49,57],work:[49,57],would:[41,44,57],wouldn:49,wrap:49,write:[44,57],wrong:[41,44,57],x64:49,x_ax:49,x_eigval:49,x_i:41,x_latent:49,xaxi:49,xlabel:49,xtick:49,xticklabel:49,xxt:49,xxt_eigval:49,yaxi:49,ylabel:49,you:[44,49,57],your:49,yticklabel:49,zoom:[]},titles:["&lt;no title&gt;","<span class=\"section-number\">3. </span>Algorithms for more than 2 graphs","<span class=\"section-number\">3.2. </span>Testing for Significant Communities","&lt;no title&gt;","<span class=\"section-number\">3.1. </span>Testing for Significant Vertices","<span class=\"section-number\">1.5. </span>Anomaly Detection","<span class=\"section-number\">1. </span>Leveraging Representations for Single Graph Applications","<span class=\"section-number\">1.1. </span>Community Detection","<span class=\"section-number\">1.3. </span>Model Selection","&lt;no title&gt;","<span class=\"section-number\">1.2. </span>Testing for Differences between Communities","<span class=\"section-number\">1.4. </span>Vertex Nomination","<span class=\"section-number\">2. </span>Leveraging Representations for Multiple Graph Applications","<span class=\"section-number\">2.2. </span>Graph Matching and Vertex Nomination","<span class=\"section-number\">2.1. </span>Two-Sample Hypothesis Testing","<span class=\"section-number\">2.3. </span>Vertex Nomination","<span class=\"section-number\">1. </span>The Network Data Science Landscape","<span class=\"section-number\">1.3. </span>Examples of applications","<span class=\"section-number\">1.6. </span>Exercises","<span class=\"section-number\">1.5. </span>Types of Network Learning Problems","<span class=\"section-number\">1.4. </span>Types of Networks","<span class=\"section-number\">1.1. </span>What Is A Network?","<span class=\"section-number\">1.2. </span>Why Study Networks?","<span class=\"section-number\">2. </span>End-to-end Biology Network Data Science Project","<span class=\"section-number\">2.1. </span>Discover and Visualize the Data to Gain Insights","<span class=\"section-number\">2.5. </span>Fine-Tune your Model","&lt;no title&gt;","<span class=\"section-number\">2.2. </span>Prepare the Data for Network Algorithms","<span class=\"section-number\">2.4. </span>Select and Train a Model","<span class=\"section-number\">2.3. </span>Transformation Techniques","<span class=\"section-number\">3.1. </span>Look at the Big Picture","<span class=\"section-number\">3. </span>End-to-end Business Network Data Science Project","<span class=\"section-number\">3.3. </span>Discover and Visualize the Data to Gain Insights","<span class=\"section-number\">3.2. </span>Get the Data","<span class=\"section-number\">3.4. </span>Prepare the Data for Network Algorithms","Network Machine Learning in Python","<span class=\"section-number\">1. </span>Properties of Networks as a Statistical Object","<span class=\"section-number\">1.1. </span>Matrix Representations","<span class=\"section-number\">1.2. </span>Network Representations","<span class=\"section-number\">1.3. </span>Properties of Networks","<span class=\"section-number\">1.4. </span>Regularization","<span class=\"section-number\">2. </span>Why Use Statistical Models?","Network Models with Covariates","Multi-Network Models","Single-Network Models","Why Use Statistical Models?","<span class=\"section-number\">3. </span>Learning Graph Representations","<span class=\"section-number\">3.1. </span>Estimating Parameters in Network Models","<span class=\"section-number\">3.4. </span>Graph Neural Networks","<span class=\"section-number\">3.6. </span>Joint Representation Learning","<span class=\"section-number\">3.5. </span>Multigraph Representation Learning","<span class=\"section-number\">3.3. </span>Random-Walk and Diffusion-based Methods","<span class=\"section-number\">3.2. </span>Why embed networks?","<span class=\"section-number\">4. </span>Theoretical Results","<span class=\"section-number\">4.3. </span>Theory for Graph Matching","<span class=\"section-number\">4.2. </span>Theory for Multiple-Network Models","<span class=\"section-number\">4.1. </span>Theory for Single Network Models","Single-Network Models"],titleterms:{"case":49,"erd\u00f6":[44,57],"r\u00e9nyi":[44,57],The:16,Use:[41,45],Using:49,about:[],algorithm:[1,27,34],alpha:49,anomali:5,applic:[6,12,17],approach:[],aren:41,assist:49,author:[],base:51,better:49,between:10,big:30,biologi:23,block:[44,49,57],busi:31,care:41,commun:[2,7,10],compar:41,correct:[44,57],covari:[42,49],data:[16,23,24,27,31,32,33,34],degre:[44,57],detect:[5,7],differ:10,diffus:51,discov:[24,32],dot:[44,57],edg:[44,57],emb:52,embed:49,end:[23,31],equat:49,erdo:[44,57],estim:47,exampl:17,exercis:18,featur:[],fine:25,gain:[24,32],gener:[44,57],get:[33,49],good:49,graph:[1,6,12,13,44,46,48,54,57],graspolog:49,grdpg:[44,57],hypothesi:14,ier:[44,57],independ:[44,57],inhomogen:[44,57],insight:[24,32],joint:49,landscap:16,learn:[19,35,46,49,50],leverag:[6,12],look:30,machin:35,match:[13,54],matrix:37,mean:49,method:51,model:[8,25,28,41,42,43,44,45,47,49,55,56,57],more:1,multi:43,multigraph:50,multipl:[12,55],network:[16,19,20,21,22,23,27,31,34,35,36,38,39,41,42,43,44,47,48,52,55,56,57],neural:48,nomin:[11,13,15],object:36,other:[],our:49,paramet:47,pictur:30,prefac:[],prepar:[27,34],prerequisit:[],problem:19,product:[44,57],project:[23,31],properti:[36,39],python:35,random:[41,44,51,57],rang:49,rdpg:[44,57],refer:[44,49,57],regular:40,renyi:[44,57],represent:[6,12,37,38,46,49,50],resourc:[],result:53,right:41,roadmap:[],sampl:14,sbm:[44,57],scienc:[16,23,31],search:49,select:[8,28],set:49,siem:[44,57],signific:[2,4],singl:[6,44,56,57],spectral:49,statist:[36,41,45],stochast:[44,49,57],structur:[44,57],studi:22,techniqu:29,test:[2,4,10,14],than:1,theoret:53,theori:[54,55,56],train:28,transform:29,tune:25,two:14,type:[19,20],univari:41,variat:49,vertex:[11,13,15],vertic:4,visual:[24,32],walk:51,weight:49,what:21,why:[22,41,45,52],world:[],your:25}})