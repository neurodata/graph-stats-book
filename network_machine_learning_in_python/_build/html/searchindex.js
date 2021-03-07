Search.setIndex({docnames:["applications/ch10/anomaly-detection","applications/ch10/ch10","applications/ch10/significant-communities","applications/ch10/significant-edges","applications/ch10/significant-vertices","applications/ch8/anomaly-detection","applications/ch8/ch8","applications/ch8/community-detection","applications/ch8/model-selection","applications/ch8/out-of-sample","applications/ch8/testing-differences","applications/ch8/vertex-nomination","applications/ch9/ch9","applications/ch9/graph-matching-vertex","applications/ch9/two-sample-hypothesis","applications/ch9/vertex-nomination","foundations/ch1/ch1","foundations/ch1/examples-of-applications","foundations/ch1/exercises","foundations/ch1/types-of-learning-probs","foundations/ch1/types-of-networks","foundations/ch1/what-is-a-network","foundations/ch1/why-study-networks","foundations/ch2/ch2","foundations/ch2/discover-and-visualize","foundations/ch2/fine-tune","foundations/ch2/get-the-data","foundations/ch2/prepare-the-data","foundations/ch2/select-and-train","foundations/ch2/transformation-techniques","foundations/ch3/big-picture","foundations/ch3/ch3","foundations/ch3/discover-and-visualize","foundations/ch3/get-the-data","foundations/ch3/prepare-the-data","intro","representations/ch4/ch4","representations/ch4/matrix-representations","representations/ch4/network-representations","representations/ch4/properties-of-networks","representations/ch4/regularization","representations/ch5/ch5","representations/ch5/models-with-covariates","representations/ch5/multi-network-models","representations/ch5/single-network-models","representations/ch5/why-use-models","representations/ch6/ch6","representations/ch6/graph-neural-networks","representations/ch6/joint-representation-learning","representations/ch6/multigraph-representation-learning","representations/ch6/random-walk-diffusion-methods","representations/ch6/why-embed-networks","representations/ch7/ch7","representations/ch7/theory-matching","representations/ch7/theory-multigraph","representations/ch7/theory-single-network"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["applications/ch10/anomaly-detection.ipynb","applications/ch10/ch10.ipynb","applications/ch10/significant-communities.ipynb","applications/ch10/significant-edges.ipynb","applications/ch10/significant-vertices.ipynb","applications/ch8/anomaly-detection.ipynb","applications/ch8/ch8.ipynb","applications/ch8/community-detection.ipynb","applications/ch8/model-selection.ipynb","applications/ch8/out-of-sample.ipynb","applications/ch8/testing-differences.ipynb","applications/ch8/vertex-nomination.ipynb","applications/ch9/ch9.ipynb","applications/ch9/graph-matching-vertex.ipynb","applications/ch9/two-sample-hypothesis.ipynb","applications/ch9/vertex-nomination.ipynb","foundations/ch1/ch1.ipynb","foundations/ch1/examples-of-applications.ipynb","foundations/ch1/exercises.ipynb","foundations/ch1/types-of-learning-probs.ipynb","foundations/ch1/types-of-networks.ipynb","foundations/ch1/what-is-a-network.ipynb","foundations/ch1/why-study-networks.ipynb","foundations/ch2/ch2.ipynb","foundations/ch2/discover-and-visualize.ipynb","foundations/ch2/fine-tune.ipynb","foundations/ch2/get-the-data.ipynb","foundations/ch2/prepare-the-data.ipynb","foundations/ch2/select-and-train.ipynb","foundations/ch2/transformation-techniques.ipynb","foundations/ch3/big-picture.ipynb","foundations/ch3/ch3.ipynb","foundations/ch3/discover-and-visualize.ipynb","foundations/ch3/get-the-data.ipynb","foundations/ch3/prepare-the-data.ipynb","intro.md","representations/ch4/ch4.ipynb","representations/ch4/matrix-representations.ipynb","representations/ch4/network-representations.ipynb","representations/ch4/properties-of-networks.ipynb","representations/ch4/regularization.ipynb","representations/ch5/ch5.ipynb","representations/ch5/models-with-covariates.ipynb","representations/ch5/multi-network-models.ipynb","representations/ch5/single-network-models.ipynb","representations/ch5/why-use-models.ipynb","representations/ch6/ch6.ipynb","representations/ch6/graph-neural-networks.ipynb","representations/ch6/joint-representation-learning.ipynb","representations/ch6/multigraph-representation-learning.ipynb","representations/ch6/random-walk-diffusion-methods.ipynb","representations/ch6/why-embed-networks.ipynb","representations/ch7/ch7.ipynb","representations/ch7/theory-matching.ipynb","representations/ch7/theory-multigraph.ipynb","representations/ch7/theory-single-network.ipynb"],objects:{},objnames:{},objtypes:{},terms:{"00000795":46,"00019536957141698985":48,"00043756":46,"00090108":48,"0009010833022217993":48,"004155975322328694":48,"005625861749316":48,"00562586174932":48,"017044913788664812":48,"017126149668054324":48,"017149083444858545":48,"017294208713800992":48,"019168184381196746":48,"027389678814754834":48,"07680":48,"08186263117598762":48,"08318":48,"08319":48,"08336":48,"08342":48,"08600743004374037":48,"08840747693989694":48,"09025788904156264":48,"09136419645543331":48,"09161704113449845":48,"09337":48,"104":[46,48],"1093":[46,48],"12426":48,"129":[46,48],"137":[46,48],"148":48,"149":48,"150":48,"1500":[46,48],"151":48,"152":48,"175":48,"176":48,"177":48,"178":48,"179":48,"180":48,"1936":[46,48],"1959":44,"1982":[46,48],"1ebf56eac912":48,"20105":48,"2017":[46,48],"22945":48,"235931277557181e":48,"2359312775571826e":48,"24767":48,"24805":48,"24852":48,"26147":48,"28080":48,"282":48,"283":48,"284":48,"285":48,"286":48,"290":44,"297":44,"3144":44,"321":[46,48],"32529":48,"361":[46,48],"377":[46,48],"37789":48,"40775285877078155":48,"44406":48,"47159":48,"52632":48,"60732":48,"673872836847387":48,"69900":48,"70282":48,"70563":48,"70729":48,"70832":48,"70897":48,"70932":48,"70956":48,"70969":48,"70978":48,"72484":48,"83676":48,"85159":48,"8581568507e8":44,"8806372446165007":48,"case":44,"class":[46,48],"float":[46,48],"function":[46,48],"import":[44,46,48],"new":[44,46,48],"return":[46,48],"short":44,"true":[44,46,48],"try":[46,48],Being:44,But:48,For:[44,46,48],One:[46,48],The:[44,46,48],Then:48,There:[46,48],These:[44,46,48],Using:44,_emb:48,_fit_transform:48,_get_tuning_paramet:48,_ll:48,_xxt:48,abl:[46,48],about:[44,46,48],abov:[44,46,48],access:[46,48],accomplish:[46,48],achiev:48,actual:[46,48],added:48,addit:[46,48],adj:44,adjac:[44,46,48],after:[46,48],against:[46,48],age:[44,46,48],agreement:[44,46,48],algebra:[46,48],algorithm:[46,48],align:[44,46,48],all:[44,48],allow:48,almost:[44,48],alpha:46,alpha_:[46,48],alreadi:[44,48],also:[44,46,48],alwai:44,amax:[46,48],amin:[46,48],amount:48,analysi:[44,46,48],analyt:44,ani:44,anoth:44,answer:44,anticip:44,anyth:44,aphor:44,appar:44,appli:[46,48],appropri:44,arbitrari:44,aren:48,arg:48,argument:48,arr:44,arrai:[44,46,48],arrang:44,ascend:48,assign:44,associ:[46,48],assort:[46,48],assum:[44,48],asx008:[46,48],attent:44,attribut:44,avail:[46,48],averag:44,axes:48,axessubplot:[44,46,48],axi:[46,48],axs:[46,48],axx:[46,48],base:48,basi:44,becaus:[44,48],becom:44,befor:[44,48],begin:[44,46,48],behav:44,behavior:44,being:44,believ:44,belong:[46,48],below:[44,46,48],bern:[46,48],bernoulli:[44,46,48],best:[46,48],best_alpha:[46,48],between:[44,46,48],big:[46,48],binkiewicz:[46,48],biomet:[46,48],biometrika:[46,48],bit:48,black:[46,48],bmatrix:[46,48],both:[46,48],box:[44,46,48],brain:[46,48],call:[44,46,48],can:[44,46,48],candid:44,capabl:[46,48],care:44,casc:[46,48],cbar:[46,48],cdot:[46,48],center:[44,46,48],chanc:44,chang:[46,48],check:[46,48],choic:[44,46,48],clarifi:44,classic:[46,48],clearli:[44,48],close:44,cluster:[46,48],cmap:[46,48],code:[44,46,48],collect:[46,48],color:[46,48],colorbar:[46,48],column:48,combin:[46,48],come:[46,48],common:44,commun:[44,46,48],complex:44,complic:[46,48],comput:44,computation:[46,48],concat:[46,48],conceiv:[46,48],concern:44,connect:[44,48],consid:44,constrained_layout:[46,48],contain:[46,48],context:44,contribut:[46,48],convei:44,correctli:[46,48],correl:[46,48],correspond:44,could:[44,46,48],covariateassistedembed:[46,48],covariateassistedspectralembed:[46,48],cover:44,crappi:46,creat:48,custom:[46,48],dad:[46,48],dark:44,data:[44,48],datafram:46,dataset:[46,48],deal:44,debrecen:44,decompos:[46,48],decomposit:[46,48],deduc:44,def:[46,48],defin:[44,46,48],definit:48,deg:44,delin:44,denot:44,depend:[44,46],deprec:44,describ:[44,46,48],determin:44,develop:44,devis:44,df1:46,df2:46,diag_indices_from:48,diagon:44,dict:46,did:[46,48],differ:[44,46,48],difficult:[44,46,48],dimens:[46,48],dimension:[46,48],direct:[44,48],discern:44,displai:[44,48],distanc:[46,48],distinct:[44,46,48],distinguish:48,distribut:[44,48],doe:[44,48],doesn:[46,48],doi:[46,48],don:[46,48],dot:[46,48],down:[46,48],draw:48,drop:46,dropbox:48,due:44,each:[44,46,48],earlier:[46,48],easili:44,edg:[46,48],effect:44,effici:[44,46,48],eigenvalu:[46,48],eigenvector:46,eigvalsh:[46,48],either:44,elif:46,els:48,emb:[46,48],embedding_alg:[46,48],emphas:[46,48],end:[44,46,48],enough:48,enti:44,entri:44,equal:[46,48],equat:46,equival:44,er_n:44,er_np:44,erestim:44,error:44,essenti:[46,48],estim:44,evalu:48,even:[44,46,48],everi:[44,46,48],everyth:46,exact:44,exactli:[46,48],exampl:[44,46,48],exce:44,exceed:48,exercis:44,exist:44,expect:44,expens:[46,48],explain:44,exploit:44,explor:[46,48],extra:[46,48],extract:46,fact:44,factor:44,fairli:[46,48],fall:44,fals:[44,46,48],famili:44,faster:[46,48],few:[46,48],fig:[46,48],figsiz:[46,48],figur:[46,48],filterwarn:[46,48],find:[46,48],fine:48,finit:44,first:[44,46,48],firstli:46,fit:[44,46,48],fit_transform:[46,48],fix:[46,48],flip:[46,48],follow:[44,46,48],fomal:44,form:[46,48],formal:44,fortun:[46,48],frac:[46,48],friend:44,from:[44,46,48],from_list:[46,48],front:44,full:48,fundament:44,further:44,futur:44,futurewarn:44,game:48,gca:48,gen_covari:[46,48],gen_sbm:48,gender:[46,48],gener:48,geomspac:48,georg:44,get:44,get_eigv:48,give:[46,48],given:44,glue:48,goal:[46,48],good:44,govern:44,graph:48,graspolog:44,greater:[46,48],group:[46,48],had:44,hand:44,has:[44,48],have:[44,46,48],heatmap:[44,46,48],help:48,henceforth:[46,48],here:[44,46,48],higher:[44,46,48],highest:48,hold:44,holist:[46,48],hotel:[46,48],how:[44,48],howev:48,html:48,http:[46,48],hue:[46,48],hypothet:44,ideal:44,ident:44,ieee:[46,48],ignor:[46,48],illustr:[44,48],impact:44,imperfect:48,implement:[46,48],imposs:48,improv:[46,48],incid:44,includ:44,increas:[44,48],inde:44,index:[44,46],indistinguish:48,inertia:[46,48],inertia_:[46,48],inferenti:44,influenc:44,inform:[44,46,48],initi:48,input:[44,48],instanc:[44,46,48],instanti:44,instead:[44,48],interest:44,interpret:[44,46,48],intuit:44,intuition:44,invert:44,investig:[44,46,48],ipython:[44,48],issu:[46,48],iter:[46,48],its:[44,46,48],itself:48,jointli:[46,48],june:[46,48],just:[46,48],keep:46,kei:[46,48],kmean:[46,48],know:44,known:[44,46,48],kwarg:48,l_ax:[46,48],l_eigval:[46,48],l_latent:48,l_top:48,label:[46,48],lack:44,lambda:[46,48],lambda_1:[46,48],lambda_:[46,48],lambda_i:[46,48],lambda_k:[46,48],lambda_r:[46,48],laplacian:[46,48],laplacianspectralemb:48,larger:[44,46,48],last:48,latent:[46,48],latent_left_:48,latent_posit:[46,48],latent_right_:48,latents_:48,later:[44,46,48],layer:48,lead:[46,48],learn:44,least:[46,48],left:[44,48],len:[46,48],length:48,less:[44,46,48],let:[44,46,48],level:44,like:[44,46,48],linalg:[46,48],line:[46,48],linear:[46,48],linearsegmentedcolormap:[46,48],linewidth:[46,48],linspac:46,lloyd:[46,48],locat:[46,48],longer:48,look:[44,46,48],loop:44,lower:48,lse:48,luck:48,machin:48,mai:44,make:[44,46,48],manag:48,mani:[44,46,48],math:[44,46,48],mathbb:44,mathcal:44,matplotlib:[46,48],matric:[44,46,48],matrix:[44,46,48],max:[46,48],maxfun:48,maximum:[46,48],mean:44,measur:[46,48],member:44,merit:44,method:48,might:[44,46,48],min:[46,48],minibatchkmean:46,minimum:[46,48],modul:48,more:[44,46,48],most:[46,48],mtx:44,much:[46,48],multidimension:44,must:44,myst_nb:48,n_cluster:[46,48],n_compon:[46,48],n_covari:[46,48],n_eigval:48,n_vertic:48,name:46,natur:[46,48],ncol:[46,48],ndd:48,need:[44,46,48],neq:44,neuron:48,never:44,new_lat:48,nice:[46,48],nit:48,node:[44,46,48],nois:[46,48],non:[44,46,48],none:48,nor:44,normal:48,note:44,noth:44,notic:[46,48],now:[46,48],nrow:[46,48],num:[46,48],number:[44,46,48],numpi:[44,46,48],obei:44,object:[44,46,48],observ:44,obviou:44,occur:44,off:[44,46],often:[46,48],old:44,one:[44,48],ones:[46,48],onli:[44,46,48],operand:48,optimizeresult:48,option:[46,48],order:[44,48],org:[46,48],organ:[46,48],origin:[46,48],other:[44,46,48],otherwis:[44,48],our:[44,46],out:[46,48],overlai:48,overlap:[46,48],own:[44,46,48],page:[46,48],pai:44,pair:44,pairplot:[46,48],palett:[46,48],panda:46,paper:[46,48],paramet:[44,46,48],parlanc:[46,48],particular:[44,46,48],particularli:44,pattern:44,pcm:[46,48],peopl:44,per:48,perf_count:46,perfectli:[46,48],permut:44,person:[46,48],physic:48,pick:[46,48],piec:44,place:[44,46,48],plai:48,plot:[44,46,48],plot_lat:[46,48],plt:[46,48],pmb:44,point:[46,48],posit:[46,48],possess:44,possibl:[44,46,48],potenti:44,practic:44,preced:44,predetermin:48,preprocess:48,present:[44,48],pretti:46,previous:48,primari:[44,46,48],principl:48,print:[44,46,48],prior:48,probabl:[44,46,48],problem:[46,48],procedur:[44,46,48],process:44,produc:[44,46,48],product:[46,48],promis:48,properti:44,provid:[44,46,48],publ:44,put:46,pyplot:[46,48],python:[44,46,48],quantiz:[46,48],question:44,quicker:48,quickli:[46,48],quit:44,random:[46,48],randomli:44,rather:44,ratio:[46,48],realiz:44,reason:[44,46,48],recal:44,recent:48,red:44,reduc:[46,48],refin:44,regardless:44,region:[46,48],regular:[46,48],rel:44,relat:[46,48],remaind:44,rememb:[46,48],reorder:44,replac:44,repres:48,reset_index:46,resort:44,respect:[44,46],result:[44,46,48],retriev:[46,48],return_label:[46,48],revers:48,right:[44,48],rohe:[46,48],role:44,row:[46,48],rvs:[46,48],same:[44,46,48],sampl:44,sbm:[46,48],sbm_n:44,scatterplot:[46,48],school:[44,48],scientif:44,scientist:44,scikit:[46,48],scipi:[46,48],seaborn:[46,48],second:[44,46,48],section:[44,46,48],see:[44,48],seed:48,seem:[44,46,48],select:44,selectsvd:[46,48],self:48,sens:44,separ:48,seq:44,sequenc:44,set1:[46,48],set:44,set_frame_on:[46,48],set_tick:[46,48],set_ticklabel:[46,48],set_titl:[46,48],set_vis:48,shape:[46,48],share:48,should:[46,48],show:[44,46,48],shown:[46,48],sim:44,similar:[44,46,48],similarli:[44,46,48],simpl:[44,46,48],simpler:48,simplest:44,simpli:[44,46,48],simplic:[44,48],simul:[44,46,48],sinc:[46,48],singl:[46,48],situat:[44,46,48],size:[44,46,48],sklearn:[46,48],slower:48,small:44,smaller:[46,48],sns:[46,48],social:[44,46,48],solut:48,some:[44,46,48],somehow:[46,48],someth:44,sometim:[46,48],somewhat:[46,48],somewher:48,spars:44,specif:46,spend:[46,48],squar:[44,46,48],stabl:48,standard:[46,48],start:[44,46,48],stat:[46,48],state:44,statist:44,stick:48,str:46,straightforward:[44,46,48],structur:[46,48],structuur:44,struuctur:44,stuart:[46,48],student:44,studi:44,subgraph:44,subplot:[46,48],subset_by_index:48,sum:[44,46,48],sum_:44,suppos:[44,48],svd:[46,48],symmetr:44,symmetri:44,take:[44,46,48],task:44,tau:[44,46,48],tau_i:44,techniqu:[44,46,48],tell:[46,48],ten:48,tend:44,test:48,textrm:44,than:[44,46,48],thei:[46,48],them:46,theoret:44,theori:[44,46,48],thi:[44,46,48],thing:[44,46],third:[46,48],three:[46,48],through:[46,48],thte:44,tight_layout:48,time:[44,46,48],titl:[44,46,48],to_laplacian:[46,48],todo:[46,48],togeth:[46,48],too:44,tool:[46,48],top:48,top_eigv:48,total:[44,48],traceback:48,tractabl:44,transact:[46,48],transpos:[46,48],trick:48,trivial:44,truncat:48,tune:[46,48],tuning_rang:[46,48],tupl:[44,48],tutori:48,two:[44,46,48],type:48,typeerror:48,underli:44,understand:44,undirect:44,uninform:48,uniqu:44,unit:48,unless:44,unsupport:48,until:[46,48],updat:46,upon:44,use:[44,46,48],used:[44,48],useful:[44,46,48],uses:[46,48],using:[44,46,48],usual:[44,46],util:[46,48],v_i:44,v_j:44,valu:[44,46,48],variant:[46,48],vec:44,vector:[44,48],veri:[44,46,48],version:[46,48],vertex:44,vertic:44,vetex:44,virtual:44,visual:[44,46,48],vogelstein:[46,48],volum:[46,48],vtx_perm:44,wai:[44,46,48],want:[46,48],warn:[46,48],weight:44,were:[46,48],what:[46,48],when:[44,46,48],where:[44,46,48],wherein:44,whether:44,which:[44,46,48],white:[44,46,48],willing:48,wish:44,wite:44,within:[44,46,48],word:[44,46,48],work:[46,48],would:44,wouldn:[46,48],wrap:48,write:44,wrong:44,x_ax:[46,48],x_eigval:48,x_latent:48,xaxi:48,xlabel:48,xtick:[46,48],xxt:[46,48],xxt_eigval:[46,48],yaxi:48,ylabel:[46,48],you:[44,46,48],your:[46,48],zoom:46},titles:["&lt;no title&gt;","Algorithms for more than 2 graphs","Testing for Significant Communities","&lt;no title&gt;","Testing for Significant Vertices","Anomaly Detection","Leveraging Representations for Single Graph Applications","Community Detection","Model Selection","&lt;no title&gt;","Testing for Differences between Communities","Vertex Nomination","Leveraging Representations for Multiple Graph Applications","Graph Matching and Vertex Nomination","Two-Sample Hypothesis Testing","Vertex Nomination","The Network Data Science Landscape","Examples of applications","Exercises","Types of Network Learning Problems","Types of Networks","What Is A Network?","Why Study Networks?","End-to-end Biology Network Data Science Project","Discover and Visualize the Data to Gain Insights","Fine-Tune your Model","&lt;no title&gt;","Prepare the Data for Network Algorithms","Select and Train a Model","Transformation Techniques","Look at the Big Picture","End-to-end Business Network Data Science Project","Discover and Visualize the Data to Gain Insights","Get the Data","Prepare the Data for Network Algorithms","Preface","Properties of Networks as a Statistical Object","Matrix Representations","Network Representations","Properties of Networks","Regularization","Why Use Statistical Models?","Network Models with Covariates","Multi-Network Models","Single-Network Models","Why Use Statistical Models?","Learning Graph Representations","Graph Neural Networks","Network and Vertex feature joint representation learning","Multigraph Representation Learning","Random-Walk and Diffusion-based Methods","Why embed networks?","Theoretical Results","Theory for Graph Matching","Theory for Multiple-Network Models","Theory for Single Network Models"],titleterms:{"case":[46,48],"erd\u00f6":44,"r\u00e9nyi":44,The:[16,35],Use:[41,45],Using:[46,48],about:35,algorithm:[1,27,34],alpha:48,anomali:5,applic:[6,12,17],approach:35,assist:[46,48],author:35,base:50,better:[46,48],between:10,big:30,biologi:23,block:[44,46,48],busi:31,commun:[2,7,10],correct:44,covari:[42,46,48],data:[16,23,24,27,31,32,33,34,46],degre:44,detect:[5,7],differ:10,diffus:50,discov:[24,32],dot:44,edg:44,emb:51,embed:[46,48],end:[23,31],equat:48,erdo:44,exampl:17,exercis:18,featur:[46,48],fine:25,gain:[24,32],gener:[44,46],get:[33,46,48],good:48,graph:[1,6,12,13,44,46,47,53],graspolog:[46,48],grdpg:44,hypothesi:14,ier:44,independ:44,inhomogen:44,insight:[24,32],joint:[46,48],landscap:16,learn:[19,46,48,49],leverag:[6,12],look:30,match:[13,53],matrix:37,mean:[46,48],method:50,model:[8,25,28,41,42,43,44,45,46,48,54,55],more:1,multi:43,multigraph:49,multipl:[12,54],network:[16,19,20,21,22,23,27,31,34,35,36,38,39,42,43,44,46,47,48,51,54,55],neural:47,nomin:[11,13,15],object:[35,36],other:35,our:48,pictur:30,prefac:35,prepar:[27,34],prerequisit:35,problem:19,product:44,project:[23,31],properti:[36,39],random:[44,50],rang:[46,48],rdpg:44,refer:[44,46,48],regular:40,renyi:44,represent:[6,12,37,38,46,48,49],resourc:35,result:52,roadmap:35,sampl:14,sbm:44,scienc:[16,23,31],search:48,select:[8,28],set:[46,48],siem:44,signific:[2,4],singl:[6,44,55],spectral:[46,48],statist:[36,41,45],stochast:[44,46,48],structur:44,studi:22,techniqu:29,test:[2,4,10,14],than:1,theoret:52,theori:[53,54,55],train:28,transform:29,tune:25,two:14,type:[19,20],variat:[46,48],vertex:[11,13,15,46,48],vertic:4,visual:[24,32],walk:50,weight:[46,48],what:21,why:[22,41,45,51],world:35,your:25}})