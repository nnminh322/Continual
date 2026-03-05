

---

# ENHANCED ANALYSIS WITH PDF INTRODUCTIONS

## Paper 01: Gated Integration of Low-Rank Adaptation for CL of Large Language Models

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
1.0
0.5
0.0 Large XL
ModelSize
)M(sretemaraPelbaniarT 3.5
3.0
```

## Paper 02: MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
t MoE static
g α
tasksdisagreeontheirbestexpert.
```

## Paper 03: MemEIC: A Step Toward Continual and Compositional Knowledge Editing

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
VisRel(%) TextRel(%) CompRel(%)
(a)LLaVA-1.5
(b)MiniGPT-4
TrainingStage2: KnowledgeConnector Inthesecondstage,weactivateboththevisualand
textual adapters and train the Knowledge Connector on the CCKEB training examples using an
adversarialretriever. Insteadofdirectlyrelyingonexternalmemory,theretrieverprovidesamixture
of factual and counterfactual evidence to simulate realistic and noisy retrieval conditions. The
connector is optimized to fuse the outputs of the two adapters only when a compositional query
requirestheintegrationofbothvisualandtextualknowledge,therebydiscouragingover-relianceon
externalmemoryandencouragingselectiveintegrationofinternalandexternalevidence.
Testing: For Realistic Editing In deployment, users Train (Stage 1, Training External Memory)
rarelymodifybothvisualandtextualfactssimultaneously; Input Deco Q m u p e o ry sition E M x e t m er o n r a y l Base LLM
rather,editsondifferentmodalitiesoftenoccur—forex-
Train (Stage 2, TrainingConnector)
ample,atextualeditmaylaterbefollowedbyavisualedit
asnewinformationbecomesavailable. Hence,wefreeze Input Deco Q m u p e o ry sition A R d e v t e r r ie sa ve ri r al T V e i x s t u u a a l l a a d d a a p p t t e e r r Cecotnonr B L a L s M e
theKnowledgeConnector;whenanedittargetsinternal Counterfactual Retrieval
(Wrong)
knowledge,weupdatethecorrespondingvisualortextual Factual Retrieval
Retrievals (Correct) Edit Space
adapter,whilenewevidenceissimplyappended(stacked) Test
intheexternalmemorystore,leavingprior
```

## Paper 04: Investigating and Mitigating CF in Medical Knowledge Injection through Internal Knowledge Augmentation Learning

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
Investigating and Mitigating Catastrophic Forgetting
in Medical Knowledge Injection through Internal
Knowledge Augmentation Learning
YuxuanZhou1,XienLiu∗1,XiaoZhang1,ChenNing1,ShijinWang2,GuopingHu2,JiWu1,3,4
1DepartmentofElectronicEngineering,TsinghuaUniversity,Beijing,China
2iFLYTEKResearch,Hefei,China 3CollegeofAI,TsinghuaUniversity,Beijing,China
4BeijingNationalResearchCenterforInformationScienceandTechnology,Beijing,China
Abstract
LargeLanguageModels(LLMs)areexpectedtopossesscomprehensivemedical
knowledge to support real-world clinical applications. While domain-specific
fine-tuningeffectivelyinjectsmedicalknowledgeintoLLMs,itoftencausescatas-
trophicforgettingofpreviouslyacquiredknowledgeandinstruction-followingca-
pabilities. Inthispaper,weinvestigatethisissueandrevealapatternofproximity-
dependent forgetting: knowledge that is semantically or topically close to the
injectedcontentismorelikelytobeforgotten,whileunrelatedknowledgeshows
minimaldegradation. Moreover,weobservethatexistingmitigationtechniques
failtoaddressthistypeofforgettingeffectively. Motivatedbythisobservation
and inspired by human learning mechanisms, we propose InternAL (Internal
KnowledgeAugmentationLearning),anovelapproachthatleveragesLLMs’own
internalknowledgetomitigateforgetting. InternALfirstprobesinternalknowledge
closely related to the injection by prompting the model with questions derived
fromtheinjectedknowledge. Thisknowledgeisthenusedtoaugmenttheoriginal
injectiondataset,guidingthemodeltor
```

## Paper 05: Bisecle: Binding and Separation in CL for Video Language Understanding

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
Binding Separation Bisecle: Binding and Separation in Continual
Continual Learning
Learning for Video Language Understanding
23
YueTan XiaoqianHu
SchoolofComputerScience SchoolofComputerScience
UniversityofNewSouthWales UniversityofNewSouthWales
Sydney,Australia Sydney,Australia
yue.tan@unsw.edu.au xiaoqian.hu@student.unsw.edu.au
HaoXue CelsoDeMelo
SchoolofComputerScienceandEngineering DEVCOMArmyResearchLaboratory
UniversityofNewSouthWales USA
Sydney,Australia celso.miguel.de.melo@gmail.com
hao.xue1@unsw.edu.au
FloraD.Salim∗
SchoolofComputerScienceandEngineering
UniversityofNewSouthWales
Sydney,Australia
flora.salim@unsw.edu.au
Abstract
Frontiervision-languagemodels(VLMs)havemaderemarkableimprovementsin
videounderstandingtasks. However,real-worldvideostypicallyexistascontinu-
ouslyevolvingdatastreams(e.g.,dynamicscenescapturedbywearableglasses),
necessitatingmodelstocontinuallyadapttoshiftingdatadistributionsandnovel
scenarios. Consideringtheprohibitivecomputationalcostsoffine-tuningmodels
on new tasks, usually, a small subset of parameters is updated while the bulk
of the model remains frozen. This poses new challenges to existing continual
learningframeworksinthecontextoflargemultimodalfoundationmodels,i.e.,
catastrophicforgettingandupdateconflict. Whilethefoundationmodelsstrug-
gle with parameter-efficient continual learning, the hippocampus in the human
brain has evolved highly efficient mechanisms for memory formation and con-
solidation. InspiredbytherapidBindingandpatt
```

## Paper 06: Continual Multimodal Contrastive Learning

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
Continual Multimodal Contrastive Learning
XiaohaoLiu XiaoboXia∗ See-KiongNg Tat-SengChua
NationalUniversityofSingapore
xiaohao.liu@u.nus.edu {xbx, seekiong, dcscts}@nus.edu.sg
Abstract
Multimodal Contrastive Learning (MCL) advances in aligning different modal-
ities and generating multimodal representations in a joint space. By leveraging
contrastive learning across diverse modalities, large-scale multimodal data en-
hances representational quality. However, a critical yet often overlooked chal-
lengeremains: multimodaldataisrarelycollectedinasingleprocess,andtrain-
ingfromscratchiscomputationallyexpensive.Instead,emergentmultimodaldata
canbeusedtooptimizeexistingmodelsgradually,i.e.,modelsaretrainedonase-
quence of modality pair data. We define this problem as Continual Multimodal
Contrastive Learning (CMCL), an underexplored yet crucial research direction
at the intersection of multimodal and continual learning. In this paper, we for-
mulate CMCL through two specialized principles of stability and plasticity. We
theoretically derive a novel optimization-based method, which projects updated
gradients from dual sides onto subspaces where any gradient is prevented from
interfering with the previously learned knowledge. Two upper bounds provide
theoretical insights on both stability and plasticity in our solution. Beyond our
theoretical contributions, we conduct experiments on multiple datasets by com-
paringourmethodagainstadvancedcontinuallearningbaselines. Theempirical
resul
```

## Paper 07: Demystifying Language Model Forgetting with Low-rank Example Associations

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
Weuselogperplexityasthemainperformancemetricasitisapplicabletobothlanguagemodeling
andinstructiontuning,andisknowntocorrelatewellwithotherdataset-specificmetrics(Hoffmann
etal.,2022). Forinstructiontuningtaskswitharestrictedoutputspace(e.g.,multi-choicequestions),
wealsomeasurebinaryexact-matchaccuracy(EM).Wemeasureforgettingz thatoccursonan
ij
upstreamexamplex ∈x asincreaseinlogperplexityordropinexactmatchafterfine-tuning
j 1..N
theLMonanewtaskT ∈T . Werecordforgettingz inanassociationmatrixZ ∈RM×N.
i 1..M ij
```

## Paper 08: Turning the Tables: Enabling Backward Transfer via Causal-Aware LoRA in CL

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
Turning the Tables: Enabling Backward Transfer via
Causal-Aware LoRA in Continual Learning
ChaoyangLi1,2 RunzeYe1 JianyangQin1 JinhaoCui1
LingzhiWang1 NingHu2
QingLiao1,2∗
1HarbinInstituteofTechnology,Shenzhen,China
2PengChengLaboratory,Shenzhen,China
{22b951022, 24S151081, 22b351005, cuijinhao}@stu.hit.edu.cn,
{hun}@pcl.ac.cn,{wanglingzhi, liaoqing}@hit.edu.cn
Abstract
Currentparameter-efficientfine-tuning(PEFT)methodshaveshownsuperiorper-
formanceincontinuallearning.However,mostexistingPEFT-basedmethodsfocus
onmitigatingcatastrophicforgettingbylimitingmodificationstotheoldtaskmodel
causedbynewtasks.Thishindersbackwardknowledgetransfer,aswhennewtasks
have a strong positive correlation with old tasks, appropriately training on new
taskscantransferbeneficialknowledgetooldtasks. Critically,achievingbackward
knowledgetransferfacestwofundamentalchallenges: (1)someparametersmay
beineffectiveontaskperformance,whichconstrainsthetasksolutionspaceand
modelcapacity;(2)sinceoldtaskdataareinaccessible,modelingtaskcorrelation
viashareddataisinfeasible. Toaddressthesechallenges,weproposeCaLoRA,a
novelcausal-awarelow-rankadaptationframeworkthatisthefirstPEFT-based
continuallearningworkwithbackwardknowledgetransfer. Specifically,wefirst
proposeparameter-levelcounterfactualattribution(PaCA)thatestimatesthecausal
effectofLoRAparametersviacounterfactualreasoning,identifyingeffectivepa-
rametersfromacausalview. Second,weproposecross-taskgradientadaptation
(CaGA)toquantifytaskcorrelationbygradien
```

## Paper 09: Mitigating Intra- and Inter-modal Forgetting in CL of Unified Multimodal Models

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
additiontothemorecommonintra-modalforgettinginautoregressivetransformersthatunify
multimodalunderstandingandgenerationundercontinualinstructiontuning. Wefurther
provideatheoreticalanalysisattributinginter-modalforgettingtomodalitygradientconflict,
andformallyprovethatmodalitydecouplingcanmitigatethisconflictboththeoretically
andexperimentally.
```

## Paper 10: MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs

**Venue:** NeurIPS | **Year:** 2025
**arxiv:** N/A

### Introduction (from PDF)
```
desirable balance between reliability, 0.8 0.6
0.4 0.6
generalization, and locality for a large 0.2 0.0
number of edits (Figure 1). MEMOIR 0.4
introduces a memory module, i.e., a Loc. Gen. 0.2
dedicatedfully-connectedlayerinasingle 0.0
transformer block where all edits are T=1000 10 500 1000 1500 2000 2500 3000
T
performed. MEMOIRmitigatescatastrophic
MEMIT GRACE WISE AlphaEdit MEMOIR
forgettingbyallocatingdistinctparameter
subsets to each edit and retrieving them
Figure1:Left:trade-offsamongreliability(Rel.),gener-
during inference to activate only the
alization(Gen.),andlocality(Loc.) for1,000continual
relevant knowledge for a given prompt.
edits. Right: AverageperformanceofRel.,Gen.,and
During editing, we propose a structured
Loc. undervaryingnumbersofedits. Bothevaluated
sparsification of the input with sample-
onLLaMA-3-8B-Instruct[16]withZsREdataset[17].
dependentmaskstothislayer,dynamically
MEMOIRdeliversthebestbalanceamongthethreemet-
activating only a sample-specific subset
ricsandscaleswithalargenumberofedits.
of parameters in the introduced memory
module in the forward pass. MEMOIR
therefore promotes the distribution of new knowledge across the parameter space, reducing the
overwritingofpreviousupdatesandsignificantlymitigatingcatastrophicforgetting. Duringinference,
we use the sparsification pattern of a prompt to infer if it semantically corresponds to an edited
prompt, androutetheactivationsaccordingly. Thistargetedknowledgeactivationeliminatesthe
needforlargecorporaofirrelevantsamplestoregularizetraining[14,15,9]. Specifically, ifthe
inputcorrespondstoarephrasedversionofanedit,weactivateonlytherelevantknowledgeofthat
edittoaligntherepresentationoftheprompt. Conversely,theintroducedmoduleisdeactivatedwhen
detectingirrelevantprompts,effectivelypreservingthepre-trainedknowledgeoftheLLM.
Ourcontributionsareasfollows:
• WeproposeMEMOIR,anovellifelongmodeleditingmethodtocontinuouslyeditalongsequence
ofsampleswithminimalforgettingviasparselydistributingknowledgeinthememorymodule
fromeachedit. Duringinference,MEMOIRidentifieseditedsamples,theirrephrasedcounterparts,
andirrelevantsamplestodynamicallyactivateordeactivaterelevantknowledge.
• WeextensivelyevaluateMEMOIRonQ&A,hallucinationcorrection,OODgeneralization,and
multi-hopreasoningtasks,demonstratingstate-of-the-artresultsacrossLLaMA-3[16],Mis-
tral[18],LLaMA-2[19]andGPT-J[20]architecturescomparedtoallpreviousmethods.
2
• Weextendthepreviousedithorizonto15,000editsandshowthatMEMOIRconsistentlydelivers
superior performance in the challenging setting of sequential singular edits compared with
previouseditingmethods.
2 LifelongModelEditing
Problem formulation Lifelong model editing aims to continuously add new knowledge into
amodelwithoutforgettingpreviouslyacquiredknowledge[8,9]. Letθ ∈ Rd betheparameters
of an LLM denoted as a function f : X → Y. It maps an input prompt x ∈ X to an output
θ
predictionf (x)∈Y. TheLLMhasbeentrainedwithinternet-scalecorporaD . Duringediting,
θ train
the model receives a time-evolving stream of edit samples D = {(xt,yt)} , where each pair
edit e e t
(xt,yt) ∈ D represents the t-th edit sample in the edit sequence. In real-world settings, this
e e edit
sequenceisexpectedtogrowcontinually,potentiallyreachingthousandsofedits.
When performing the t-th edit, the goal of lifelong model editing is to incorporate the new edit
while preserving previously acquired knowledge. Formally, for the parameters of the t-th edit
θt, our objectiveis for theupdated model topredict y = f (x), ∀(x,y) ∈ D ∪D≤t where
θt train edit
D≤t :={(xs,ys)} . Inpractice,D isoftenunavailableandweapproximatetheretentionof
edit e e s≤t train
pre-trainedknowledgebyapredefinedsetofsamplesirrelevanttotheeditdataset: (x,y)∈D .
irr
KnowledgeeditingforLLMs LLMsaretypicallybasedonthetransformerarchitecture[21]and
consist of L identical blocks containing multi-head attention modules followed by feed-forward
network(FFN)modules. Foreachblockℓ∈[L],theFFNmoduleconsistsoftwofullyconnected
layers, parameterized by Wℓ and Wℓ , and an activation function. Previous work shows that
fc proj
knowledgeistypicallystoredinthemiddleblocksofanLLM[11],andupdatingtheparametersofa
singlemiddleblockratherthanfine-tuningtheentiremodelcanbeeffectivetoeditfactualknowledge
[9,8].Inwhatfollows,tolightennotation,wewillomitthelayer-indexingsuperscript.Lethrepresent
theoutputfromtheattentionmodule;thentheoutputoftheFFNmoduleisformallydefinedas:
FFN(h)=W σ(W h), (1)
proj fc
foranelement-wiseactivationfunctionσ(·). TheFFNmodulecanbeinterpretedasatwo-layer
key-value memory system [22], where the second layer W functions as a linear associative
proj
memory [11], mapping a sequence of input vectors [a ,a ,...] to corresponding output vectors
1 2
[v ,v ,...]. Thismemory-likebehaviorenablesinformationretrieval,wherea = σ(W h)and
1 2 fc
v =FFN(h)respectivelyrepresenttheinputandoutputvectorsofthelayerW . Leveragingthis
proj
perspective,previousworkshaveshowntheeffectivenessofmodifyingtheoutputofW toupdate
proj
themodel’sstoredknowledge[11,22,9,14,15]. However,continuouslyupdatingW inlifelong
proj
editinggraduallyoverwritespreviousedits,leadingtosignificantforgettingandsevereperformance
degradationwithalargenumberofedits,asshowninFigure1.
3 MemoryEditingwithMinimalOverwriteandInformedRetention
We propose MEMOIR: Model Editing with Minimal Overwrite and Informed Retention. During
editing,MEMOIRdistributesknowledgestoragebyallocatingdistinctparametersubsetsforeacheditto
reduceoverwritingofpreviousedits(Section3.1). Atinferencestage,MEMOIRdynamicallyactivates
relevantparametersrelatedtotheknowledgeofeachinputprompt(Section3.2). Throughthisdesign,
MEMOIRachievesasuperiorbalanceamongreliability,generalization,andlocalityevenwithalarge
numberofedits. WestartbypresentingthegeneralframeworkofMEMOIRillustratedinFigure2.
Editinglayeroutputwithresidualmemory Followingpriorworks[11,15],wesolelymodify
theoutputofaprojectionlayeroftheFFNW ofaspecificblockℓ,whichisdenotedasW in
proj 0
thefollowing. Topreservetheknowledgestoredinthepre-trainedmodel,wedonotdirectlymodify
thepre-trainedweightsofthislayerbutintroducearesidualmemorylayerW toincorporatenew
m
edits. W isazero-initializedcopyoftheoriginalmatrixW suchthatitcontainsnoinformation
m 0
beforeediting. Thefinaloutputoftheeditedlayercombinestheoutputoftheoriginallayerand
theresidualmemorylayer.
3
Input query:x<latexit sha1_base64="Dym7DfStHtyfn6V1ZhDoz0V+jDI=">AAAB9XicbVC7TsMwFL3hWcqrwMhiUSExVQniNVZiYSwSfUhtqBzHaa06dmQ7QBX1P1gYQIiVf2Hjb3DaDNByJMtH59wrH58g4Uwb1/12lpZXVtfWSxvlza3tnd3K3n5Ly1QR2iSSS9UJsKacCdo0zHDaSRTFccBpOxhd5377gSrNpLgz44T6MR4IFjGCjZXue4HkoR7H9sqeJv1K1a25U6BF4hWkCgUa/cpXL5QkjakwhGOtu56bGD/DyjDC6aTcSzVNMBnhAe1aKnBMtZ9NU0/QsVVCFElljzBoqv7eyHCs82h2MsZmqOe9XPzP66YmuvIzJpLUUEFmD0UpR0aivAIUMkWJ4WNLMFHMZkVkiBUmxhZVtiV4819eJK3TmndRO789q9bdoo4SHMIRnIAHl1CHG2hAEwgoeIZXeHMenRfn3fmYjS45xc4B/IHz+QNRIpMB</latexit>
Get activations Timestep <latexit sha1_base64="adV5rEF4rAUzZaNKjl9V7C9+8y0=">AAAB6nicbVDLSgNBEOyNrxhfUY9eBoPgKeyKr4sQ8OIxonlAsoTZyWwyZHZ2mekVwpJP8OJBEa9+kTf/xkmyB00saCiquunuChIpDLrut1NYWV1b3yhulra2d3b3yvsHTROnmvEGi2Ws2wE1XArFGyhQ8naiOY0CyVvB6Hbqt564NiJWjzhOuB/RgRKhYBSt9IA3Xq9ccavuDGSZeDmpQI56r/zV7ccsjbhCJqkxHc9N0M+oRsEkn5S6qeEJZSM64B1LFY248bPZqRNyYpU+CWNtSyGZqb8nMhoZM44C2xlRHJpFbyr+53VSDK/9TKgkRa7YfFGYSoIxmf5N+kJzhnJsCWVa2FsJG1JNGdp0SjYEb/HlZdI8q3qX1Yv780rNzeMowhEcwyl4cAU1uIM6NIDBAJ7hFd4c6bw4787HvLXg5DOH8AfO5w/PtY1y</latexit>t=1 <latexit sha1_base64="BbKVvgioCizyykV0TuMFq2T2GUo=">AAAB6nicbVDLSgNBEOz1GeMr6tHLYBA8hd3g6yIEvHiMaB6QLGF2MpsMmZ1dZnqFsOQTvHhQxKtf5M2/cZLsQRMLGoqqbrq7gkQKg6777aysrq1vbBa2its7u3v7pYPDpolTzXiDxTLW7YAaLoXiDRQoeTvRnEaB5K1gdDv1W09cGxGrRxwn3I/oQIlQMIpWesCbaq9UdivuDGSZeDkpQ456r/TV7ccsjbhCJqkxHc9N0M+oRsEknxS7qeEJZSM64B1LFY248bPZqRNyapU+CWNtSyGZqb8nMhoZM44C2xlRHJpFbyr+53VSDK/9TKgkRa7YfFGYSoIxmf5N+kJzhnJsCWVa2FsJG1JNGdp0ijYEb/HlZdKsVrzLysX9ebnm5nEU4BhO4Aw8uIIa3EEdGsBgAM/wCm+OdF6cd+dj3rri5DNH8AfO5w/ROY1z</latexit>t=2 . <latexit sha1_base64="wjZPxDMQAAMqDMn6wOUZ8MrRuIc=">AAAB6nicbVBNS8NAEJ3Ur1q/qh69LBbBU0jEr2PBi8eK1hbaUDbbSbt0swm7G6GE/gQvHhTx6i/y5r9x2+agrQ8GHu/NMDMvTAXXxvO+ndLK6tr6RnmzsrW9s7tX3T941EmmGDZZIhLVDqlGwSU2DTcC26lCGocCW+HoZuq3nlBpnsgHM04xiOlA8ogzaqx077pur1rzXG8Gskz8gtSgQKNX/er2E5bFKA0TVOuO76UmyKkynAmcVLqZxpSyER1gx1JJY9RBPjt1Qk6s0idRomxJQ2bq74mcxlqP49B2xtQM9aI3Ff/zOpmJroOcyzQzKNl8UZQJYhIy/Zv0uUJmxNgSyhS3txI2pIoyY9Op2BD8xZeXyeOZ61+6F3fntbpXxFGGIziGU/DhCupwCw1oAoMBPMMrvDnCeXHenY95a8kpZg7hD5zPH0m6jRo=</latexit> ..
<latexit sha1_base64="YhNExUIfsSJoYKCoghTHEqmOnzU=">AAAB+3icbVC7TsMwFL0pr1JeoYwsFhUSU5UgXmMlFsYi0VKpjSLHcVurjhPZDmoV5VdYGECIlR9h429w2gzQciTLR+fcKx+fIOFMacf5tipr6xubW9Xt2s7u3v6BfVjvqjiVhHZIzGPZC7CinAna0Uxz2kskxVHA6WMwuS38xycqFYvFg54l1IvwSLAhI1gbybfrgyDmoZpF5sqmuZ+5uW83nKYzB1olbkkaUKLt21+DMCZpRIUmHCvVd51EexmWmhFO89ogVTTBZIJHtG+owBFVXjbPnqNTo4RoGEtzhEZz9fdGhiNVxDOTEdZjtewV4n9eP9XDGy9jIkk1FWTx0DDlSMeoKAKFTFKi+cwQTCQzWREZY4mJNnXVTAnu8pdXSfe86V41L+8vGi2nrKMKx3ACZ+DCNbTgDtrQAQJTeIZXeLNy68V6tz4WoxWr3DmCP7A+fwDJ1ZTi</latexit>x1 <latexit sha1_base64="Pxmg9ATfz3UKEYW/hpjEDFyas9I=">AAAB+3icbVA7T8MwGHTKq5RXKCOLRYXEVCUVr7ESC2OR6ENqo8hxnNaqY0e2g1pF+SssDCDEyh9h49/gtBmg5STLp7vvk88XJIwq7TjfVmVjc2t7p7pb29s/ODyyj+s9JVKJSRcLJuQgQIowyklXU83IIJEExQEj/WB6V/j9JyIVFfxRzxPixWjMaUQx0kby7fooECxU89hc2Sz3s1bu2w2n6SwA14lbkgYo0fHtr1EocBoTrjFDSg1dJ9FehqSmmJG8NkoVSRCeojEZGspRTJSXLbLn8NwoIYyENIdruFB/b2QoVkU8MxkjPVGrXiH+5w1THd16GeVJqgnHy4eilEEtYFEEDKkkWLO5IQhLarJCPEESYW3qqpkS3NUvr5Neq+leN68eLhttp6yjCk7BGbgALrgBbXAPOqALMJiBZ/AK3qzcerHerY/laMUqd07AH1ifP8talOM=</latexit>x2
Get activations Get activations
Conditional knowledge
activation TopHash TopHash
Edit knowledge Edit knowledge
Original Residual
memory memory
Residual memory Residual memory Residual memory
Edited output
(a) Overview of MEMOIR (b) Knowledge distribution for residual memory during editing
Figure2: (a)OverallframeworkofMEMOIRduringinferencestage. Theeditedoutputcombinesthe
outputsoftheoriginallayeroutputandtheresidualmemory. Theinputtotheresidualmemorycondi-
tionallyactivatesspecificcolumnsintheresidualmemorytoretrieverelevantknowledge. (b)During
editing,eacheditmodifiesonlyadesignatedsubsetofcolumnsintheresidualmemory,minimizing
overwritingofpreviouseditsinthememory. Forvisualization,wetransposetheweightmatrices.
3.1 Editing: distributingknowledgeacrossmemory
Priorparametricapproachesonknowledgeediting[14,15]typicallystoretheknowledgeofeach
editacrosstheentiretrainableparameterspace. However,unconstrainedupdatesonalltrainable
parametersleadtoseverecatastrophicforgetting,asneweditsquicklyoverwriteknowledgefrom
previousones,asshowninFigure1. Inthiswork,wedrawmotivationfromthecontinuallearning
literatureleveragingsparsity[23,24,25,26,27,28]tomitigatecatastrophicforgetting,andpropose
astructuredsparsificationofinputactivationstotheresidualmemoryduringtheforwardpass. This
ensuresthateacheditmodifiesonlya subsetofthetrainableparameters, promoting less overlap
inrepresentationsandleadingtomorelocalizedupdatesacrossdifferentedits.
Concretely,foraninputpromptxlettheinputactivationsoftheeditedlayerbea(x)∈RD,where
weomitthetokendimensionforclarity. WeapplyabinarymaskM:RD (cid:55)→{0,1}D totheinput
activations,andthefinaloutputoftheeditedlayer,denotedasFFN (a(x)),iscalculatedas:
edited
FFN (a(x))=W a(x)+W (M(a(x))⊙a(x)), (2)
edited 0 m
Applying the sparse mask M(a(x)) to the input activations retains only k ≪ D active indices,
settingtheresttozero. Asaresult,gradientupdatesarerestrictedtothekcorrespondingcolumns
inW ,wheretheknowledgeforinputxisexplicitlystored. Theothercolumnsremainunchanged,
m
effectivelypreservingpreviouslystorededitsandminimizinginterference. Inpractice,weconstruct
themaskbasedontheactivationaveragedacrossalltokensinthepromptxandthenapplyituniformly
totheactivationofeachtoken. Tosuppressfeatureswithconsistentlyhighmagnitudeacrossall
prompts,theaverageactivationsarecenteredbyremovingtheirmeancomputedwith100irrelevant
prompts.Thisenablesamorediverseselectionofmasksacrossprompts.Weprovidemoredetailsand
ablationsinAppendixB.2. Next,wedescribehowthemaskisderivedusingtheTopHashmechanism.
TopHash Afundamentalaspectofthissparsemaskingistheselectionmechanismfortheactive
indices. Intuitively,thismechanismshouldsatisfytwodifferentcriteria: selectionoffeaturediversity
andsemanticcoherence. Diversityinfeatureselectionencouragesthemodeltoconsiderabroader
set of features, reducing the risk of overfitting on dominant features and mitigating catastrophic
forgettingbydistributingupdatesacrossthetrainableparametersofthememorymodule. Semantic
coherenceensuresthatsemanticallysimilarinputpromptsyieldsimilarselectedmasks,facilitating
theactivationofrelevantknowledgeinthememorymoduleforpreviouslyunseenyetsemantically
```

