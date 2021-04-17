Search.setIndex({docnames:["applications/ch10/anomaly-detection","applications/ch10/ch10","applications/ch10/significant-communities","applications/ch10/significant-edges","applications/ch10/significant-vertices","applications/ch8/anomaly-detection","applications/ch8/ch8","applications/ch8/community-detection","applications/ch8/model-selection","applications/ch8/out-of-sample","applications/ch8/testing-differences","applications/ch8/vertex-nomination","applications/ch9/ch9","applications/ch9/graph-matching-vertex","applications/ch9/two-sample-hypothesis","applications/ch9/vertex-nomination","foundations/ch1/ch1","foundations/ch1/examples-of-applications","foundations/ch1/exercises","foundations/ch1/main-challenges","foundations/ch1/types-of-learning-probs","foundations/ch1/types-of-networks","foundations/ch1/what-is-a-network","foundations/ch1/why-study-networks","foundations/ch2/big-picture","foundations/ch2/ch2","foundations/ch2/discover-and-visualize","foundations/ch2/fine-tune","foundations/ch2/get-the-data","foundations/ch2/prepare-the-data","foundations/ch2/select-and-train","foundations/ch2/transformation-techniques","foundations/ch3/big-picture","foundations/ch3/ch3","foundations/ch3/discover-and-visualize","foundations/ch3/get-the-data","foundations/ch3/prepare-the-data","intro","representations/ch4/ch4","representations/ch4/matrix-representations","representations/ch4/network-representations","representations/ch4/properties-of-networks","representations/ch4/regularization","representations/ch5/ch5","representations/ch5/models-with-covariates","representations/ch5/multi-network-models","representations/ch5/single-network-models","representations/ch5/why-use-models","representations/ch6/ch6","representations/ch6/estimating-parameters","representations/ch6/graph-neural-networks","representations/ch6/joint-representation-learning","representations/ch6/multigraph-representation-learning","representations/ch6/random-walk-diffusion-methods","representations/ch6/why-embed-networks","representations/ch7/ch7","representations/ch7/theory-matching","representations/ch7/theory-multigraph","representations/ch7/theory-single-network","representations/single-network-models"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinxcontrib.bibtex":7,sphinx:56},filenames:["applications/ch10/anomaly-detection.ipynb","applications/ch10/ch10.ipynb","applications/ch10/significant-communities.ipynb","applications/ch10/significant-edges.ipynb","applications/ch10/significant-vertices.ipynb","applications/ch8/anomaly-detection.ipynb","applications/ch8/ch8.ipynb","applications/ch8/community-detection.ipynb","applications/ch8/model-selection.ipynb","applications/ch8/out-of-sample.ipynb","applications/ch8/testing-differences.ipynb","applications/ch8/vertex-nomination.ipynb","applications/ch9/ch9.ipynb","applications/ch9/graph-matching-vertex.ipynb","applications/ch9/two-sample-hypothesis.ipynb","applications/ch9/vertex-nomination.ipynb","foundations/ch1/ch1.ipynb","foundations/ch1/examples-of-applications.ipynb","foundations/ch1/exercises.ipynb","foundations/ch1/main-challenges.ipynb","foundations/ch1/types-of-learning-probs.ipynb","foundations/ch1/types-of-networks.ipynb","foundations/ch1/what-is-a-network.ipynb","foundations/ch1/why-study-networks.ipynb","foundations/ch2/big-picture.ipynb","foundations/ch2/ch2.ipynb","foundations/ch2/discover-and-visualize.ipynb","foundations/ch2/fine-tune.ipynb","foundations/ch2/get-the-data.ipynb","foundations/ch2/prepare-the-data.ipynb","foundations/ch2/select-and-train.ipynb","foundations/ch2/transformation-techniques.ipynb","foundations/ch3/big-picture.ipynb","foundations/ch3/ch3.ipynb","foundations/ch3/discover-and-visualize.ipynb","foundations/ch3/get-the-data.ipynb","foundations/ch3/prepare-the-data.ipynb","intro.md","representations/ch4/ch4.ipynb","representations/ch4/matrix-representations.ipynb","representations/ch4/network-representations.ipynb","representations/ch4/properties-of-networks.ipynb","representations/ch4/regularization.ipynb","representations/ch5/ch5.ipynb","representations/ch5/models-with-covariates.ipynb","representations/ch5/multi-network-models.ipynb","representations/ch5/single-network-models.ipynb","representations/ch5/why-use-models.ipynb","representations/ch6/ch6.ipynb","representations/ch6/estimating-parameters.ipynb","representations/ch6/graph-neural-networks.ipynb","representations/ch6/joint-representation-learning.ipynb","representations/ch6/multigraph-representation-learning.ipynb","representations/ch6/random-walk-diffusion-methods.ipynb","representations/ch6/why-embed-networks.ipynb","representations/ch7/ch7.ipynb","representations/ch7/theory-matching.ipynb","representations/ch7/theory-multigraph.ipynb","representations/ch7/theory-single-network.ipynb","representations/single-network-models.ipynb"],objects:{},objnames:{},objtypes:{},terms:{"100":[46,51,52],"104":51,"105":46,"1093":51,"10e":51,"121":52,"122":52,"129":51,"137":51,"149":46,"150":46,"1500":51,"1936":51,"1959":[46,59],"1982":51,"200":46,"2017":51,"250":46,"290":[46,59],"297":[46,59],"299":46,"300":46,"302":59,"321":51,"350":46,"361":51,"377":51,"500":51,"8581568507e8":59,"case":[43,46,51,52,59],"class":[43,51,59],"final":[46,51,52],"function":[46,51],"import":[43,46,51,52,59],"int":46,"new":[43,46,51,59],"return":[46,51,52],"short":46,"super":52,"true":[46,51,52,59],"try":[46,52],"while":46,ASE:52,But:[51,52],Doing:52,For:[43,46,51,52,59],Not:52,One:46,That:46,The:[43,46,51,59],Then:52,There:[46,51,52],These:[43,46,51,59],Use:52,Using:[46,59],With:[46,51,52],_get_omni_matrix:52,a_hm:52,a_n:46,abl:[46,51,52],abnorm:52,abnormal_mean:52,abnormal_plot_gmm:52,about:[43,46,51,52,59],abov:[43,46,51,52,59],absurdli:46,access:[46,51],accord:[46,52],account:43,accur:59,accuraci:43,achiev:51,across:[46,52],act:51,activ:52,actual:[43,46,51,52,59],add:[46,52],add_ax:52,added:[46,51],adding:43,addit:[43,46,51],address:[43,46],adj:[46,59],adjac:[46,51,59],adjacencyspectralemb:52,adjplot:[46,52],advanc:[46,51],affect:[46,59],after:[46,51,52],again:[43,46],age:[43,51,59],agreement:51,ahead:46,algebra:51,algorithm:[51,52,59],alic:46,align:[46,51,52],all:[43,46,51,52,59],all_lat:52,allow:[43,46,51],almost:[51,59],alpha:[51,52],alreadi:[46,51,59],also:[43,46,51,52],altern:46,alwai:[46,51,59],alzheim:52,amount:[43,46,51],analog:46,analysi:51,ani:[43,46,51,52,59],annoi:51,anoth:[43,46,51,52,59],answer:[43,46,59],anybodi:52,anyth:46,aphor:[43,59],appar:59,appear:46,append:52,appli:[46,51],applic:43,approach:[43,52,59],appropri:[43,59],arang:46,arbitrari:[43,46],aren:52,around:43,arr:59,arrai:[46,51,52,59],arrang:[46,59],arrow:52,arrow_ax:52,ase:52,ask:46,aspect:[43,59],assign:[46,59],associ:51,assum:[43,46,59],assumpt:[43,46,52],asterisk:46,astyp:46,asx008:51,attempt:46,attend:46,attende:46,attent:[43,59],attribut:[43,46,59],averag:[46,52,59],axes:[51,52],axes_al:52,axes_grid1:52,axes_pad:52,axessubplot:[46,52,59],axi:[51,52],axj:52,axs:[46,51,52],ayi:51,background:46,ballpark:46,base:52,basi:[43,59],bay:46,becaus:[43,46,51,52,59],becom:[46,52,59],befor:[46,51,52,59],begin:[43,46,51],behav:[46,59],behavior:59,being:[43,46,51,59],believ:[46,59],belong:[51,52],below:[46,51,52,59],ben:37,bern:46,bernoulli:[43,46,59],best:[46,51,52,59],beta:51,better:51,between:[46,51,52,59],betwen:46,bia:43,big:46,biggest:51,binari:51,binary_heatmap:[46,52],binkiewicz:51,binom:46,binomi:46,biolog:46,biomet:51,biometrika:51,bit:[46,51],black:[46,51,52],block:52,blue:46,bmatrix:51,bob:46,book:[43,46],both:[46,51,52],bother:43,bottom:52,box:[43,59],brain:[51,52],breadth:43,british:[43,59],build:46,built:46,bummer:52,bunch:[46,52],call:[43,46,51,52,59],came:43,can:[43,46,51,52,59],candid:[43,59],cap:46,capac:46,captur:[43,46,52],cardin:46,care:[46,59],carefulli:46,casc:51,categor:46,caus:51,cax:52,cbar:[46,51,52],cbar_ax:52,cbar_kw:46,cdot:[46,51],center:[46,51,52,59],centuri:[43,59],chadwick:46,chanc:[43,46,51,52,59],chang:52,chapter:52,character:43,characterist:51,cheat:51,check:[46,51,59],choic:[46,59],choos:59,chosen:51,circl:43,claim:46,clarifi:[43,59],classif:52,clear:52,clearli:[46,51,52,59],close:[43,46],cluster:[46,51,52],cmap:[46,51,52],code:[51,52,59],coin:43,col_nam:46,collect:[43,46,51],color:[46,51,52],colorbar:[46,51,52],column:[46,51],comb:46,combin:[46,51],come:[43,46,51,52],commmun:51,common:[52,59],commonli:51,commun:[43,46,51,52,59],compar:52,comparison:51,complet:[46,51],complex:[43,46,52,59],complic:46,compon:[46,51],compris:46,comput:[46,52,59],concaten:52,concept:[43,46],conclud:46,condit:46,configur:46,connect:[43,46,51,52,59],connectom:37,consid:[43,46,59],consist:[43,46,51,52],constrained_layout:51,construct:46,contact:43,contain:51,content:46,context:[43,51,59],contrast:46,contribut:51,control:51,convei:[43,59],convolut:43,core:46,correct:[43,51],correctli:[43,51],correspond:[46,52,59],cosi:52,could:[43,46,51,52,59],count:46,counter:46,counti:43,covariateassistedemb:51,covariateassistedspectralembed:51,cover:[43,46],creat:[46,51,52],credit:37,critic:46,crucial:[43,46],current:51,custom:51,dad:51,danc:43,dark:[46,59],darker:51,dash:52,dat:46,data:[43,46,51,52,59],datafram:46,dataset:43,deal:[46,52,59],debrecen:[46,59],decompos:51,decomposit:51,deduc:46,deeper:52,def:[46,51,52],defin:[46,59],definit:[46,59],deg:46,degre:51,delet:43,delin:[46,59],denot:[46,59],depend:[46,52,59],deprec:59,depth:46,describ:[43,46,51,59],descript:[43,46],design:59,despin:52,despit:43,determin:[43,46,59],determinist:[43,46],develop:[43,52,59],diag:46,diagon:[46,51,59],dict:46,did:[43,46,51,52],didn:51,differ:[46,51,59],difficult:[51,59],dimens:[43,51],dimension:[46,51,52],direct:[46,59],directli:[43,51,52,59],discern:59,discov:[43,52],discret:46,discrim:43,discuss:46,diseas:52,dispar:43,displot:46,distinct:[46,51],distinguish:51,distribut:[43,46,51,52,59],dive:52,doe:[46,51,59],doesn:[43,51,59],doi:51,doing:51,domin:51,don:[43,46,51,52],dot:[51,52],doubl:46,down:[46,51],draw:51,drawn:[51,52],dual:52,due:59,e_i:46,each:[43,46,51,52,59],earlier:51,easi:46,easiest:[46,52],easili:[46,52,59],edg:[43,46,51,52],effect:[43,46,51],eigenvalu:51,eigenvector:51,eigsh:51,either:[51,52,59],elegan:37,elif:52,ell:46,els:[46,51],emb:[51,52],embed:46,embedding_alg:51,emphas:43,encount:43,end:[43,46,51,59],enough:59,enti:59,entir:[43,46,52],entri:[46,59],enumer:52,equal:[46,51],equat:51,equiv:46,equival:59,er_:46,er_n:46,er_np:[46,59],erestim:59,error:59,essenti:[46,51],establish:46,estim:[43,46,59],evalu:46,even:[43,46,51,52,59],eventu:52,ever:[46,59],everi:[43,46,51,52,59],everyth:[43,52],everywher:46,exact:59,exactli:[43,46,51,59],exampl:[43,51,52,59],exce:[46,59],exist:[43,46,59],expect:[43,52,59],explain:[46,59],explicitli:[43,46],explor:[46,52,59],express:46,extmath:51,extra:51,extrem:[43,46],facilit:43,fact:[43,46,59],factor:[43,59],fairli:[46,51,52],faithfulli:43,fall:59,fals:[46,51,52,59],famili:[43,59],familiar:[43,46],far:52,fast:46,featur:51,femal:46,few:43,fewer:46,fig:[46,51,52],fight:43,figsiz:[46,51,52],figur:[46,51,52],filterwarn:[51,52],find:[46,52,59],finit:46,first:[43,46,51,52,59],fit:[46,59],fit_predict:52,fit_transform:[51,52],fix:[43,46],flat:[46,51,52],flavor:46,flip:43,fold:46,follow:[43,46,51,59],fomal:59,font_scal:46,fontdict:52,fontsiz:[46,51,52],foral:46,form:51,formal:[46,59],format:46,fortun:[46,59],found:[51,52],four:[46,52],frac:[46,51],fraction:59,framework:59,friend:[43,46,59],friendship:59,from:[43,46,51,52,59],front:43,full:[43,46,51],further:[46,59],futur:59,futurewarn:59,game:51,gaussian:[46,52],gaussianmixtur:52,gca:[51,52],gen_covari:51,gender:51,gener:[43,51,52],geomspac:51,georg:[43,59],get:[43,46,51,59],get_legend:[51,52],giant:51,give:[46,59],given:[43,46,52,59],gmm:52,goal:[51,52,59],goe:46,going:52,gone:52,good:51,got:43,govern:[43,46,59],grade:[43,59],graph:[51,52],graspolog:[46,59],great:52,greater:51,grid1:52,grid2:52,group:[46,51,52,59],grow:[46,59],had:[43,46,52,59],half:46,hand:[46,51,59],happen:[46,51,52],has:[43,46,51,59],hat:46,have:[43,46,51,52,59],head:43,heatmap:[46,51,52,59],height:52,help:51,here:[46,51,52],heterogen:52,high:[46,52],higher:[43,46,51,59],hline:52,hma:52,hmap:[46,52],hmn:52,hobbi:43,hold:[43,46],hollow:46,homogen:46,hood:51,hope:43,hopkin:37,horizontalalign:52,horribl:52,hotel:51,how:[43,46,51,52,59],howev:[46,51,52],hspace:52,hstack:52,html:51,http:51,hue:[46,51,52],idea:[46,52,59],ident:[46,59],ieee:51,ignor:[51,52],illustr:[51,59],imagegrid:52,imagin:[43,46],impact:[43,46,59],impl:46,implement:[51,52],impli:[46,51],improv:51,imshow:52,incid:59,includ:[43,59],incorpor:43,increas:[43,46,59],inde:59,independ:[46,52],index:[46,59],indic:[46,59],indistinguish:51,individu:[43,46,52],infer:43,inferenti:46,influenc:[43,46,59],inform:[43,46,51,52,59],inner_hier_label:51,input:59,insight:43,instanc:[43,46,51,59],instanti:59,instead:[43,46,51,59],integ:46,integrand:46,interest:[43,59],interpret:[46,51,59],intim:43,intuit:[43,46,59],intuition:[46,59],investig:[51,59],ipython:59,isn:[43,51],issu:51,item:52,its:[51,59],itself:[43,46,51],john:37,joint:[46,52],joint_embed:52,jointli:51,june:51,just:[43,46,51,52,59],kde:46,keep:[46,51,52],kind:[46,51],knew:[43,46,51,59],knit:46,know:[43,46,51,52,59],knowledg:43,known:46,kwarg:52,l_ax:51,l_latent:51,label:[51,52],labels_abnorm:52,labels_norm:52,lack:43,lambda_1:51,land:43,laplacian:51,laplacianspectralemb:51,larg:[43,46,51,52],larger:[46,51,59],last:51,latent:[46,51,52],latent_posit:[51,52],latents_:51,latents_abnorm:52,latents_mas:52,latents_norm:52,latents_omni:52,later:[43,46,51,52,59],layer:51,lead:46,leading_eigval_l:51,leading_eigval_yyt:51,learn:[43,46,59],least:[51,52],left:[46,51,52],legend:[51,52],length:51,less:52,let:[43,46,51,52],level:[43,59],leverag:43,life:43,lighter:51,like:[43,46,51,52,59],limit:46,linalg:51,line:[46,51],linear:[46,51],linecolor:46,lineplot:46,linestyl:52,linewidth:[46,51],listedcolormap:[46,51,52],littl:[46,59],live:43,lloyd:51,loc:[46,51,52],locat:[51,52],log10:46,log:46,logan:46,logic:46,longer:[43,46,51],look:[43,46,51,52,59],loop:[46,59],loopless:46,lose:52,lot:[46,51,52,59],low:52,lower:[43,46,51,52],lse:51,luck:51,machin:[43,46],made:46,mai:[43,59],make:[46,51,52,59],make_commun:51,make_network:52,male:46,manag:[51,52],mani:[43,46,51],manner:46,manual:51,margin:46,mase:52,mase_ax:52,mass:46,massiv:46,math:[46,59],mathbb:46,mathbf:[43,46],mathcal:[46,59],mathemat:46,matplotlib:[46,51,52],matric:[46,51,52,59],matrix:[46,51,52,59],matter:[46,52],maximum:46,maxnloc:52,mayb:43,mean:[43,46,51,52,59],meaningless:52,measur:51,media:43,meet:43,member:[46,59],meta:46,metadata:46,method:[46,51,52],might:[43,46,51,52,59],mind:51,minimum:46,mirror:43,miss:[46,59],mistak:46,mixtur:[46,52],model:[37,52],modul:51,modular:46,more:[43,46,51,52,59],most:51,mpl_toolkit:52,mtx:[46,59],much:[43,46,51,52,59],multi:52,multidimension:59,multipleas:52,multipli:[46,51],multivari:[43,46],must:[43,46,59],n_compon:[51,52],n_k:46,nabnorm:52,name:46,natur:[46,51],ncol:[46,51,52],ncovari:51,necessarili:[46,51,52],need:[43,46,51,52,59],neg:51,nembed:52,neq:[46,59],neuron:51,never:[43,46,59],next:[43,46,51,59],nice:[51,52],nine:52,nmatrix:51,nnormal:52,node:[43,46,51,52,59],nois:[51,52],noisi:43,non:[46,59],none:[51,52],nope:52,nor:43,normal:[46,51,52],normal_mean:52,normal_plot_gmm:52,notation:46,note:[46,59],notebook:51,noth:[46,59],notic:[46,51],now:[43,46,51,52],nrow:[46,51],num:51,number:[46,51,52,59],numpi:[46,51,52,59],nwithout:51,object:52,observ:[43,46,59],obtain:46,obviou:[46,51],occur:[46,59],off:59,often:43,okai:59,old:[43,59],omit:46,omni:52,omni_ax:52,omni_embed_ax:52,omnibu:52,omnibusemb:52,one:[43,46,51,52,59],ones:[46,51,52],onli:[43,46,51,52,59],oper:[51,52],order:[43,46,59],org:51,organ:[46,51],other:[43,46,51,52,59],otherwis:59,our:[43,46,51,52,59],ourselv:[43,46],out:[51,52],outcom:43,outsid:46,over:[46,51],overal:51,overlai:51,overlap:51,own:[51,59],page:51,pai:[43,59],pair:[43,46,51,59],palett:[46,51,52],panda:46,paragraph:46,paramet:[43,46,51,52,59],parametr:46,particular:[43,46,51,52,59],particularli:46,partit:46,pattern:59,pcm:51,pedigo:37,peopl:[43,46,52,59],per:51,perfect:43,perfectli:51,perhap:43,permut:[46,59],person:[43,51],phd:37,physic:51,pi_1:46,pi_2:46,pi_:46,pi_k:46,pick:[51,52],pioneer:[43,59],place:[43,52],plai:[46,51],plot:[46,51,52,59],plot_heatmap:51,plot_lat:[51,52],plot_tau:46,plotting_context:46,plt:[46,51,52],pmb:[46,59],point:[43,46,51,52],posit:[51,52,59],possess:[43,59],possibl:[43,46,52,59],potenti:[43,46,51],pow:46,practic:[43,59],preced:[46,59],precis:43,predict:52,prefer:43,preprocess:51,present:46,pretend:46,pretti:46,previou:46,previous:46,primari:[43,51,59],primarili:51,print:59,priorit:43,prob:52,probabl:[43,46,51,52,59],problem:[46,51],procedur:[43,46,59],process:[43,46,59],prod_:46,produc:[43,51,52,59],product:51,properti:[43,46,59],proport:46,propos:59,propto:46,prove:46,provid:59,publ:[46,59],pull:51,purpos:[52,59],put:46,pyplot:[46,51,52],python:[51,52,59],quantiti:46,quantiz:51,question:[43,46,51,59],quit:[52,59],ram:59,random:51,randomized_svd:51,randomli:[46,59],rang:[51,52],rare:43,rather:[43,46,51,59],ratio:51,read:46,reader:46,real:46,realist:59,realiz:[43,46,59],realli:46,reason:[43,46],recal:46,recov:52,red:[46,59],reduc:51,refer:43,refin:[43,59],reflect:46,regardless:59,region:51,regular:51,rel:[46,59],relat:[43,51],relationship:46,relax:46,relev:46,remap_label:52,rememb:[43,46,51,52],remov:[51,52],reorder:[46,59],reorgan:46,replac:[46,59],repres:[43,46,51,52],represent:43,requir:[43,46],research:52,reshap:46,resort:59,respect:[46,52,59],rest:[43,46,52,59],result:[51,52,59],retriev:51,return_eigenvector:51,return_label:[51,52],revers:[46,51,59],rid:51,ridicul:46,right:[46,51,52],rightarrow:46,rm_tick:52,rocket_r:51,rohe:51,roughli:51,row:[46,51,52],rug:46,run:46,rvs:51,sai:[43,46,52,59],said:[43,52,59],same:[43,46,51,52,59],sampl:[46,52,59],sbm:[51,52],sbm_:46,sbm_n:[46,59],scale:[46,51,59],scan:52,scatterplot:[51,52],scenario:[43,46],school:[43,46,59],scienc:[43,59],scientif:[43,59],scientist:59,scikit:[51,52],scipi:51,scope:46,seaborn:[46,51,52],second:[43,46,51,52,59],section:[43,46,51,52,59],see:[43,46,51,52,59],seed:51,seek:43,seem:[46,51,52,59],select:[43,46,59],selectsvd:52,self:46,sens:[43,46,52,59],sensibl:46,separ:[46,51,52],seq:59,sequenc:59,seri:46,serv:52,set1:[51,52],set:[43,46,51,52,59],set_axis_label:46,set_frame_on:[46,51],set_major_loc:52,set_tick:[46,51,52],set_ticklabel:[46,51,52],set_titl:[46,51,52],set_vis:[51,52],set_xlabel:46,set_xtick:46,set_xticklabel:46,set_ylabel:46,set_ytick:46,set_yticklabel:46,setup:51,sever:[43,46],sex:46,shape:46,share:[46,51,52,59],share_al:52,sharei:52,sharex:52,shorthand:46,should:[46,51,52],show:[46,51,52,59],show_cbar:51,shown:46,shrink:46,signal:52,sim:[46,59],similar:[46,51,59],simpl:[43,46,51,59],simpler:[43,46,51],simplest:[46,59],simplex:46,simpli:[46,51,52,59],simplic:[43,46,59],simplifi:46,simul:[46,51,52,59],sinc:[43,46,51],singl:[43,46,51,52],singular:51,site:43,situat:[43,51,52,59],size:[46,51,59],sklearn:[51,52],skyblu:46,slightli:52,small:[46,51,59],sns:[46,51,52],social:[43,46,51,59],some:[43,46,51,52,59],somebodi:43,somehow:51,someon:43,someth:[46,59],sometim:51,somewher:51,soon:52,space:[46,51,52],spars:[46,51,59],special:[43,52],specif:[43,46],specifi:43,spectral:46,spine:52,split:46,sport:43,squar:[46,51,59],stabl:51,stack:52,standard:[51,52],start:[51,52,59],stat:51,state:[43,46,59],statement:46,statist:[46,59],statistician:[43,59],step:46,still:[43,46,52,59],stochast:[43,52],storag:46,straightforward:[46,51],strength:52,strong:52,structur:[46,51,52],structuur:59,struuctur:59,student:[37,43,46,59],stuff:52,subgraph:[46,59],subplot:[46,51,52],subplots_adjust:52,subset:46,subseteq:46,subspac:52,substanti:46,succeed:46,success:51,suit:43,sum:[46,51,59],sum_:46,summar:[43,46,59],suppos:[46,51,52,59],suptitl:[46,51,52],suspici:43,symbol:46,symmetr:[46,51,59],symmetri:59,system:43,tabl:43,tail:43,take:[43,46,51,52,59],taken:[43,46],talk:[46,52],target:43,tau:[46,51,59],tau_i:[46,59],tau_j:46,technic:[46,51,52],techniqu:[46,51,52,59],tediou:46,tell:[46,51,52],ten:46,tend:[46,59],term:[46,59],text:52,textrm:46,than:[43,46,51,59],thei:[43,46,51,52,59],them:[46,51,52],themselv:51,theorem:46,theoret:59,theori:51,therefor:[43,46,59],theta:46,thi:[43,46,51,52,59],thing:[43,46,51,52,59],think:[43,46,51,52,59],third:[51,52],thirti:51,those:[46,52],though:[51,52,59],three:[43,46,51,52],through:[46,59],throughout:46,tight_layout:[46,51,52],time:[43,46,51,59],titl:[46,51,52,59],title_fonts:52,to_laplacian:51,todo:52,togeth:[46,52,59],too:[43,46,59],tool:[51,52],top:[46,51,52],topic:52,topolog:[46,51],toss:43,total:[46,51,59],total_mean:52,tough:[46,51],track:46,tradeoff:43,tradit:[43,46],train:52,transact:51,transax:52,transform:52,transpos:[46,51],treat:46,triangleq:46,trivial:46,truncat:51,tupl:[46,59],turn:[43,46,51,52],two:[43,46,51,52,59],type:[46,51],typic:43,unambigu:46,uncertainti:43,unconnect:46,under:[46,51,59],underli:46,underneath:51,understand:[43,46,51,52],understat:46,undirect:46,undirected:46,unfortun:[43,46,51],unifi:52,uniqu:[46,59],unit:51,univari:46,univers:[37,46],unknown:46,unless:[46,59],unlik:[46,52,59],unstar:46,unsupervis:52,until:43,unweight:46,upon:59,upper:[51,52],use:[43,46,51,52,59],used:[43,46,59],useful:[43,46,51,59],uses:52,using:[43,46,51,52,59],usual:43,util:[43,51,52],v_i:[46,59],v_j:59,valu:[43,46,51,52,59],valuabl:43,variabl:[43,46],varianc:[43,46],variat:[46,51],varnoth:46,vec:[46,59],vector:[46,51,59],veri:[43,46,59],version:51,versu:46,vertex:59,vertic:59,verticalalign:52,vetex:59,viewpoint:59,virtual:59,visual:[46,51,59],vline:52,vocabulari:43,vogelstein:51,volum:51,vstack:[46,51],vtx_perm:[46,59],w_pad:52,wai:[43,46,51,52,59],want:[43,46,51,52,59],warn:[51,52],weak:52,weight:[46,59],well:[43,51,52,59],went:46,were:[43,46,51,52,59],what:[43,46,51],when:[43,46,51,52,59],whenev:46,where:[43,46,51,52,59],wherea:[43,46,52],wherein:59,whether:[43,46,52,59],which:[43,46,51,52,59],whichev:51,white:[46,51,52,59],who:46,whose:52,why:[46,59],width:52,wise:46,wish:43,wite:59,within:[46,51,52,59],without:[43,46,51,59],word:[43,46,51,59],work:[51,52,59],world:46,would:[43,46,51,52,59],wouldn:51,wrap:[51,52],write:[46,51,59],written:46,wrong:[43,46,59],wspace:52,x_ax:51,x_i:46,xaxi:[51,52],xlab:46,xlabel:[46,51],xs_1:46,xs_2:46,xtick:51,xticklabel:46,y_latent:51,yaxi:[51,52],ylabel:51,you:[46,51,52,59],your:[46,51,52],yourself:46,yticklabel:[46,51],yyt:51,zero:46,zetabyt:46,zip:[51,52]},titles:["&lt;no title&gt;","<span class=\"section-number\">3. </span>Algorithms for more than 2 graphs","<span class=\"section-number\">3.2. </span>Testing for Significant Communities","&lt;no title&gt;","<span class=\"section-number\">3.1. </span>Testing for Significant Vertices","<span class=\"section-number\">1.5. </span>Anomaly Detection","<span class=\"section-number\">1. </span>Leveraging Representations for Single Graph Applications","<span class=\"section-number\">1.1. </span>Community Detection","<span class=\"section-number\">1.3. </span>Model Selection","&lt;no title&gt;","<span class=\"section-number\">1.2. </span>Testing for Differences between Communities","<span class=\"section-number\">1.4. </span>Vertex Nomination","<span class=\"section-number\">2. </span>Leveraging Representations for Multiple Graph Applications","<span class=\"section-number\">2.2. </span>Graph Matching and Vertex Nomination","<span class=\"section-number\">2.1. </span>Two-Sample Hypothesis Testing","<span class=\"section-number\">2.3. </span>Vertex Nomination","<span class=\"section-number\">1. </span>The Network Data Science Landscape","<span class=\"section-number\">1.3. </span>Examples of applications","<span class=\"section-number\">1.7. </span>Exercises","<span class=\"section-number\">1.6. </span>Main Challenges of Network Learning","<span class=\"section-number\">1.5. </span>Types of Network Learning Problems","<span class=\"section-number\">1.4. </span>Types of Networks","<span class=\"section-number\">1.1. </span>What Is A Network?","<span class=\"section-number\">1.2. </span>Why Study Networks?","<span class=\"section-number\">2.1. </span>Look at the big picture","<span class=\"section-number\">2. </span>End-to-end Biology Network Data Science Project","<span class=\"section-number\">2.2. </span>Discover and Visualize the Data to Gain Insights","<span class=\"section-number\">2.6. </span>Fine-Tune your Model","&lt;no title&gt;","<span class=\"section-number\">2.3. </span>Prepare the Data for Network Algorithms","<span class=\"section-number\">2.5. </span>Select and Train a Model","<span class=\"section-number\">2.4. </span>Transformation Techniques","<span class=\"section-number\">3.1. </span>Look at the Big Picture","<span class=\"section-number\">3. </span>End-to-end Business Network Data Science Project","<span class=\"section-number\">3.3. </span>Discover and Visualize the Data to Gain Insights","<span class=\"section-number\">3.2. </span>Get the Data","<span class=\"section-number\">3.4. </span>Prepare the Data for Network Algorithms","Network Machine Learning in Python","<span class=\"section-number\">1. </span>Properties of Networks as a Statistical Object","<span class=\"section-number\">1.1. </span>Matrix Representations","<span class=\"section-number\">1.2. </span>Network Representations","<span class=\"section-number\">1.3. </span>Properties of Networks","<span class=\"section-number\">1.4. </span>Regularization","<span class=\"section-number\">2. </span>Why Use Statistical Models?","Network Models with Covariates","Multi-Network Models","Network Models","Why Use Statistical Models?","<span class=\"section-number\">3. </span>Learning Graph Representations","<span class=\"section-number\">3.1. </span>Estimating Parameters in Network Models","<span class=\"section-number\">3.4. </span>Graph Neural Networks","<span class=\"section-number\">3.6. </span>Joint Representation Learning","<span class=\"section-number\">3.5. </span>Multigraph Representation Learning","<span class=\"section-number\">3.3. </span>Random-Walk and Diffusion-based Methods","<span class=\"section-number\">3.2. </span>Why embed networks?","<span class=\"section-number\">4. </span>Theoretical Results","<span class=\"section-number\">4.3. </span>Theory for Graph Matching","<span class=\"section-number\">4.2. </span>Theory for Multiple-Network Models","<span class=\"section-number\">4.1. </span>Theory for Single Network Models","Single-Network Models"],titleterms:{"class":46,"erd\u00f6":[46,59],"r\u00e9nyi":[46,59],Going:52,The:[16,52],Use:[43,47],Using:[51,52],adjac:52,algorithm:[1,29,36],anomali:5,applic:[6,12,17],aren:43,assist:51,assort:51,automat:51,base:53,between:10,big:[24,32],biologi:25,block:[46,51,59],busi:33,care:43,challeng:19,classifi:52,code:46,collect:52,combin:52,commun:[2,7,10],compar:43,correct:[46,59],covari:[44,51],data:[16,25,26,29,33,34,35,36],degre:[46,59],detect:[5,7],differ:[10,52],diffus:53,discov:[26,34],dot:[46,59],edg:59,emb:54,embed:[51,52],end:[25,33],ensembl:52,equival:46,erdo:[46,59],estim:49,exampl:[17,46],exercis:[18,46],expect:46,explor:51,find:51,fine:27,foundat:46,gain:[26,34],gener:[46,59],get:35,graph:[1,6,12,13,46,48,50,56,59],graspolog:[51,52],grdpg:[46,59],hood:52,hypothesi:14,ier:[46,59],independ:59,inhomogen:[46,59],insight:[26,34],joint:51,landscap:16,learn:[19,20,37,48,51,52],leverag:[6,12],likelihood:46,look:[24,32],machin:37,main:19,match:[13,56],matrix:39,method:53,model:[8,27,30,43,44,45,46,47,49,51,57,58,59],more:1,multi:45,multigraph:52,multipl:[12,52,57],network:[16,19,20,21,22,23,25,29,33,36,37,38,40,41,43,44,45,46,49,50,51,52,54,57,58,59],neural:50,nomin:[11,13,15],non:51,object:38,out:46,paramet:49,pictur:[24,32],possibl:51,posteriori:46,practic:46,prepar:[29,36],priori:46,problem:20,product:[46,52,59],project:[25,33],properti:[38,41],python:37,random:[43,46,53,59],rdpg:[46,59],reason:51,refer:[46,51,59],regular:42,renyi:[46,59],represent:[6,12,39,40,48,51,52],result:55,right:43,sampl:14,sbm:[46,59],scienc:[16,25,33],select:[8,30],siem:59,signific:[2,4],singl:[6,58,59],spectral:[51,52],statist:[38,43,47],stochast:[46,51,59],structur:59,studi:23,techniqu:31,test:[2,4,10,14],than:1,theoret:55,theori:[56,57,58],thought:46,train:30,transform:31,tune:27,two:14,type:[20,21,52],under:52,univari:43,util:46,vertex:[11,13,15],vertic:4,visual:[26,34],walk:53,weight:51,what:[22,52],why:[23,43,47,54],work:46,your:27}})