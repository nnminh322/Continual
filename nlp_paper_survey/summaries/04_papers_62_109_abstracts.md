# Paper Abstracts: Papers 62–109

## Paper 62
**Title:** Stabilizing Zero-Shot Prediction: A Novel Antidote to Forgetting in Continual Vision-Language Tasks  
**Venue:** NeurIPS 2024

**Abstract:** Continual learning (CL) with Vision-Language Models (VLMs) like CLIP has shown promising improvements. However, existing CL methods often overlook the degradation of zero-shot prediction (ZSP) that occurs when a VLM is adapted to downstream tasks. In this work, we first analyze the causes of ZSP degradation during CL and identify two main factors: (1) the disruption of pre-trained feature representations, and (2) the interference from continuously evolving classifiers on ZSP. To address these challenges, we propose a novel approach called Zero-shot Antidote to Forgetting (ZAF). ZAF introduces a zero-shot stability regularization to preserve the pre-trained representations pivotal for ZSP. Additionally, it employs EMA-LoRA, an exponential moving average strategy applied to Low-Rank Adaptation (LoRA), to mitigate the interference of evolving classifiers on ZSP. Extensive experiments demonstrate that ZAF achieves state-of-the-art performance on benchmark datasets. Compared to the current best methods, ZAF yields average accuracy leads of 3.70% on CIFAR-100, 4.82% on ImageNet-R, and 4.38% on DomainNet, while achieving up to 10× training speed improvement.

---

## Paper 63
**Title:** Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models  
**Venue:** NeurIPS 2024

**Abstract:** Continual learning with pre-trained Vision-Language Models (VLMs) has emerged as a promising paradigm. It excels in leveraging semantically rich, well-generalized representations for incremental tasks. However, existing methods focus on preserving the pre-trained generalizability while neglecting the discriminability between pre-trained and incrementally learned knowledge. We study this issue by proposing a novel concept: cross-domain discriminability. Specifically, we reveal the challenge of classifying samples across pre-trained knowledge and new tasks due to their inherently heterogeneous representations. To address this, we propose RAIL, an approach that resolves the cross-domain discriminability issue. RAIL uses a recursive ridge regression adapter that operates on pre-trained token representations, enabling task-adaptive classification while inherently maintaining zero-shot transferability. Furthermore, we introduce X-TAIL, a new benchmark evaluating both zero-shot and few-shot continual learning across domains. The experimental results demonstrate that RAIL significantly outperforms state-of-the-art methods in X-TAIL settings, preserving zero-shot capabilities without requiring reference data.

---

## Paper 64
**Title:** Continual Learning with Global Alignment  
**Venue:** NeurIPS 2024

**Abstract:** Continual learning (CL) aims to learn a sequence of tasks incrementally without significant forgetting of previously learned knowledge. In this paper, we study a new paradigm of CL with pre-trained models by leveraging the idea of global alignment. Specifically, rather than learning an independent set of features for each new task, we compose task-specific features from a shared set of pre-trained token representations using task-specific composition weights. The correlations among tasks are naturally grounded by the pre-trained tokens. The composed task features need only be aligned with the task-specific classifier in a globally shared feature space. This design enables effective knowledge transfer and mitigates catastrophic forgetting. Our approach achieves state-of-the-art performance on standard CL benchmarks without the need for any replay buffer.

---

## Paper 65
**Title:** CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models  
**Venue:** NeurIPS 2024

**Abstract:** Continual learning with vision-language models (VLMs) aims to leverage robust pre-trained representations for incrementally arriving tasks. Existing approaches focus on preserving pre-trained zero-shot capability while learning new tasks. However, they often neglect the uncertainty and potential noise in the adaptation process. In this paper, we propose CLAP4CLIP, a probabilistic continual learning framework for CLIP-based models. Instead of learning deterministic text or visual prompt embeddings, CLAP4CLIP introduces probabilistic modeling over visual-guided text features. The framework incorporates distribution-level regularization to prevent forgetting and enables uncertainty estimation in predictions. Extensive experiments on multiple benchmarks demonstrate the effectiveness of our approach, achieving superior performance in both class-incremental and task-incremental settings.

---

## Paper 66
**Title:** Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning  
**Venue:** NeurIPS 2024

**Abstract:** Continual knowledge learning (CKL) aims to enable language models to continuously assimilate new factual knowledge while retaining previously acquired knowledge. Despite its importance, existing methods suffer from catastrophic forgetting because they treat all tokens equally during training, failing to focus on the tokens most critical for knowledge acquisition and retention. In this work, we propose Train-Attention for Language Models (TAALM), a meta-learning framework that dynamically predicts the importance weight of each token during training. TAALM uses a lightweight attention-based meta-learner to generate token-level weights that guide the training objective, emphasizing tokens most relevant to knowledge learning and retention. We also introduce LAMA-ckl, a new benchmark for evaluating CKL capability. Extensive experiments demonstrate that TAALM significantly outperforms existing baselines, reducing forgetting while improving the acquisition of new knowledge.

---

## Paper 67
**Title:** ViLCo-Bench: VIdeo Language COntinual learning Benchmark  
**Venue:** NeurIPS 2024

**Abstract:** Video-language continual learning is an important yet underexplored area of research. We present ViLCo-Bench, the first comprehensive benchmark for continual learning in video-language settings. ViLCo-Bench covers a variety of tasks including video question answering, video captioning, and video-text retrieval, constructed from ten-minute videos with diverse language queries, spanning multiple domains and requiring both temporal and spatial understanding. We evaluate several state-of-the-art continual learning methods on this benchmark and provide extensive analysis of their strengths and limitations in the video-language continual learning setting.

---

## Paper 68
**Title:** Visual Prompt Tuning in Null Space for Continual Learning  
**Venue:** NeurIPS 2024

**Abstract:** Existing prompt-based continual learning methods rely on matching input queries to a prompt pool, which is often vulnerable to cross-task interference. We propose a new approach for continual learning with Vision Transformers (ViTs) based on null-space projection of prompt gradients. Our method ensures that the gradient updates for new tasks lie in the null space of the feature space of previous tasks, thereby preventing forgetting. We develop an efficient approximation to compute the null space in the high-dimensional prompt parameter space. Experiments on multiple continual learning benchmarks demonstrate that our approach significantly outperforms prior prompt-based methods while requiring no replay buffer or task identity at inference time.

---

## Paper 69
**Title:** Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion (RAPF)  
**Venue:** ECCV 2024

**Abstract:** CLIP has demonstrated remarkable zero-shot transfer capabilities, making it a promising backbone for class-incremental learning (CIL). However, directly applying CLIP to CIL settings often leads to catastrophic forgetting of the rich pre-trained knowledge. In this paper, we propose RAPF (Representation Adjustment and Parameter Fusion), a unified framework for CLIP-based CIL. RAPF introduces adaptive representation adjustment to modify the feature representations according to new classes while minimizing interference with existing knowledge. Additionally, parameter fusion combines task-specific adapter parameters with the pre-trained model parameters in a balanced manner. Extensive experiments on standard CIL benchmarks show that RAPF achieves state-of-the-art performance while effectively preserving CLIP's zero-shot capabilities.

---

## Paper 70
**Title:** Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models (DIKI)  
**Venue:** ECCV 2024

**Abstract:** Parameter-efficient continual learning (PE-CL) of vision-language models (VLMs) has become increasingly important as it enables adapting large pre-trained models to sequential downstream tasks with minimal parameter overhead. However, existing PE-CL methods often suffer from severe interference between tasks, leading to forgetting of both pre-trained and previously learned knowledge. We present DIKI, a method that introduces a fully residual mechanism to retain pre-trained knowledge while efficiently learning new tasks. By training only 0.86% of the model's parameters, DIKI effectively mitigates interference and achieves superior performance across multiple continual learning benchmarks.

---

## Paper 71
**Title:** Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models  
**Venue:** ECCV 2024

**Abstract:** Large-scale vision-language models (VLMs) have shown a strong zero-shot generalization capability on unseen-domain data. However, adapting pre-trained VLMs to a sequence of downstream tasks often leads to the forgetting of previously learned knowledge and a reduction in zero-shot classification performance. To tackle this problem, we propose a unique Selective Dual-Teacher Knowledge Transfer framework that leverages the most recent fine-tuned and the original pre-trained VLMs as dual teachers to preserve the previously learned knowledge and zero-shot capabilities, respectively. With only access to an unlabeled reference dataset, our proposed framework performs a selective knowledge distillation mechanism by measuring the feature discrepancy from the dual-teacher VLMs. Consequently, our selective dual-teacher knowledge distillation mitigates catastrophic forgetting of previously learned knowledge while preserving the zero-shot capabilities of pre-trained VLMs. Extensive experiments on benchmark datasets demonstrate that our framework is favorable against state-of-the-art continual learning approaches for preventing catastrophic forgetting and zero-shot degradation.

---

## Paper 72
**Title:** PILoRA: Prototype Guided Incremental LoRA for Federated Class-Incremental Learning  
**Venue:** ECCV 2024

**Abstract:** Federated class-incremental learning (FCIL) tackles the challenge of learning new classes over time in a privacy-preserving federated setting. Existing methods typically struggle to balance the plasticity needed for new classes and the stability required to retain knowledge of old classes. We propose PILoRA, which leverages Low-Rank Adaptation (LoRA) guided by class prototypes for federated CIL. PILoRA uses prototype representations to efficiently condition the LoRA adapter and guide the learning of new class representations while maintaining performance on previously learned classes. The prototype-guided approach enables effective knowledge transfer across federated clients and incremental stages. Experiments demonstrate that PILoRA achieves state-of-the-art performance on federated CIL benchmarks while maintaining communication efficiency.

---

## Paper 73
**Title:** PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery  
**Venue:** ECCV 2024

**Abstract:** Continual category discovery aims to incrementally discover novel categories from unlabeled data while maintaining performance on previously discovered categories. We propose PromptCCD, which leverages a Gaussian Mixture Prompt Pool to model the distribution of visual features for both known and novel categories. The Gaussian mixture model enables soft assignment of samples to category-specific prompts, facilitating more effective category discovery. Our approach operates in a rehearsal-free manner and can adaptively expand the prompt pool as new categories emerge. Extensive experiments on benchmark datasets demonstrate that PromptCCD significantly outperforms existing methods in the continual category discovery setting.

---

## Paper 74
**Title:** Anytime Continual Learning for Open Vocabulary Classification  
**Venue:** ECCV 2024 (Oral)

**Abstract:** Open-vocabulary classification using vision-language models like CLIP has shown impressive generalization to new categories. However, adapting these models to specific domains while preserving their open-vocabulary capability remains challenging, especially in a continual learning setting. We introduce a framework for anytime continual learning that enables open-vocabulary classifiers to continuously adapt to new data and tasks at any point during deployment. Our approach supports flexible updating schedules and provides strong anytime performance guarantees. Experiments demonstrate the effectiveness of our method in maintaining both open-vocabulary generalization and task-specific performance across diverse continual learning scenarios.

---

## Paper 75
**Title:** CLIFF: Continual Latent Diffusion for Open-Vocabulary Object Detection  
**Venue:** ECCV 2024

**Abstract:** Open-vocabulary object detection (OVD) utilizes image-level cues to expand the linguistic space of region proposals, thereby facilitating the detection of diverse novel classes. Recent works adapt CLIP embedding by minimizing the object-image and object-text discrepancy combinatorially in a discriminative paradigm. However, they ignore the underlying distribution and the disagreement between the image and text objective, leading to the misaligned distribution between the vision and language sub-space. To address the deficiency, we explore the advanced generative paradigm with distribution perception and propose a novel framework based on the diffusion model, coined Continual Latent Diffusion (CLIFF), which formulates a continual distribution transfer among the object, image, and text latent space probabilistically. CLIFF consists of a Variational Latent Sampler (VLS) enabling the probabilistic modeling and a Continual Diffusion Module (CDM) for the distribution transfer. Specifically, in VLS, we first establish a probabilistic object space with region proposals by estimating distribution parameters. Then, the object-centric noise is sampled from the estimated distribution to generate text embedding for OVD. To achieve this generation process, CDM conducts a short-distance object-to-image diffusion from the sampled noise to generate image embedding as the medium, which guides the long-distance diffusion to generate text embedding. Extensive experiments verify that CLIFF can significantly surpass state-of-the-art methods on benchmarks.

---

## Paper 76
**Title:** CLEO: Continual Learning of Evolving Ontologies  
**Venue:** ECCV 2024

**Abstract:** Real-world knowledge bases and taxonomies continuously evolve over time with new categories being discovered and existing categories being refined or merged. We propose CLEO (Continual Learning of Evolving Ontologies), a framework designed to handle the dynamic evolution of category structures in continual learning settings. CLEO addresses not only the standard class-incremental learning challenge but also the restructuring of the category hierarchy over time. The method maintains a flexible representation space that supports ontology updates including category splitting, merging, and reorganization. Our experimental evaluation demonstrates CLEO's effectiveness in handling complex ontology evolution scenarios that go beyond traditional continual learning settings.

---

## Paper 77
**Title:** Exemplar-free Continual Representation Learning via Learnable Drift Compensation (LDC)  
**Venue:** ECCV 2024

**Abstract:** Continual representation learning aims to build a feature extractor that generalizes well across a sequence of tasks without storing exemplars from previous tasks. A major challenge is the representation drift—the phenomenon where the feature extractor's representations shift as it adapts to new data, causing degradation on previous tasks. We propose Learnable Drift Compensation (LDC), which introduces a lightweight, learnable module that explicitly compensates for representation drift. LDC learns to predict and correct the drift in the feature space, enabling effective knowledge retention without relying on stored exemplars. Experiments on standard continual learning benchmarks show that LDC achieves competitive or superior performance compared to exemplar-based methods while being fully exemplar-free.

---

## Paper 78
**Title:** Adapt without Forgetting: Distill Proximity from Dual Teachers in Vision-Language Models  
**Venue:** ECCV 2024

**Abstract:** Multi-modal models such as CLIP possess remarkable zero-shot transfer capabilities, making them highly effective in continual learning tasks. However, this advantage is severely compromised by catastrophic forgetting, which undermines the valuable zero-shot learning abilities of these models. Existing methods predominantly focus on preserving zero-shot capabilities but often fall short in fully exploiting the rich modal information inherent in multi-modal models. In this paper, we propose a strategy to enhance both the zero-shot transfer ability and adaptability to new data distribution. We introduce a novel graph-based multi-modal proximity distillation approach that preserves the intra- and inter-modal information for visual and textual modalities. This approach is further enhanced with a sample re-weighting mechanism, dynamically adjusting the influence of teachers for each individual sample. Experimental results demonstrate a considerable improvement over existing methodologies, which illustrate the effectiveness of the proposed method in the field of continual learning.

---

## Paper 79
**Title:** COPAL: Continual Pruning in Large Language Generative Models  
**Venue:** ICML 2024

**Abstract:** Adapting pre-trained large language models to different domains in natural language processing requires two key considerations: high computational demands and model's inability to continual adaptation. To simultaneously address both issues, this paper presents COPAL (COntinual Pruning in Adaptive Language settings), an algorithm developed for pruning large language generative models under a continual model adaptation setting. While avoiding resource-heavy finetuning or retraining, our pruning process is guided by the proposed sensitivity analysis. The sensitivity effectively measures model's ability to withstand perturbations introduced by the new dataset and finds model's weights that are relevant for all encountered datasets. As a result, COPAL allows seamless model adaptation to new domains while enhancing the resource efficiency. Our empirical evaluation on various sizes of LLMs shows that COPAL outperforms baseline models, demonstrating its efficacy in efficiency and adaptability.

---

## Paper 80
**Title:** STELLA: Continual Audio-Video Pre-training with SpatioTemporal Localized Alignment  
**Venue:** ICML 2024

**Abstract:** Audio-video pre-training has shown great promise for learning multi-modal representations. However, existing methods are typically designed for static, one-time training and struggle when new audio-video data arrives sequentially. We propose STELLA, a continual audio-video pre-training framework with spatiotemporal localized alignment. STELLA introduces a localized contrastive learning objective that aligns audio and video representations at fine-grained spatiotemporal regions, which helps maintain discriminative representations across continual learning stages. Our method achieves approximately 45% memory reduction compared to full replay approaches while maintaining competitive performance. Extensive experiments on audio-video retrieval and classification tasks demonstrate that STELLA effectively mitigates catastrophic forgetting while learning new audio-video correspondences.

---

## Paper 81
**Title:** InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning  
**Venue:** CVPR 2024

**Abstract:** Continual learning requires models to learn new tasks sequentially without forgetting previous tasks. Low-Rank Adaptation (LoRA) has shown promise for parameter-efficient fine-tuning but suffers from interference between tasks in continual learning settings. We propose InfLoRA (Interference-Free Low-Rank Adaptation), which eliminates inter-task interference by ensuring that the low-rank updates for new tasks reside in the subspace orthogonal to the important directions for previous tasks. InfLoRA achieves this through an efficient subspace decomposition that allows task-specific adaptation without compromising previous knowledge. Experiments on multiple continual learning benchmarks demonstrate that InfLoRA significantly outperforms existing LoRA-based and prompt-based continual learning methods while maintaining parameter efficiency.

---

## Paper 82
**Title:** Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters  
**Venue:** CVPR 2024

**Abstract:** Vision-language models (VLMs) like CLIP have shown remarkable generalization abilities, but adapting them to sequential tasks without forgetting remains challenging. We propose a Mixture-of-Experts (MoE) adapter framework for boosting continual learning of VLMs. Our approach dynamically routes input samples to task-relevant expert adapters, enabling specialized adaptation while preventing interference between tasks. The MoE-based design achieves a 60% reduction in trainable parameters compared to standard adapters while maintaining superior continual learning performance. Extensive experiments demonstrate that our method effectively balances the trade-off between adaptation to new tasks and preservation of previously learned knowledge and zero-shot capabilities.

---

## Paper 83
**Title:** PriViLege: Pre-trained Vision and Language transformers Are Few-Shot Incremental Learners  
**Venue:** CVPR 2024

**Abstract:** Few-shot class-incremental learning (FSCIL) aims to continuously learn new classes from limited labeled samples without forgetting old classes. We present PriViLege, which leverages pre-trained vision-language models as effective few-shot incremental learners. PriViLege exploits the rich semantic information encoded in VLMs to enable strong few-shot learning while using language-guided regularization to prevent catastrophic forgetting. Our framework demonstrates that properly harnessed VLMs can achieve state-of-the-art FSCIL performance without requiring complex architectural modifications or extensive replay mechanisms. Experiments across multiple FSCIL benchmarks confirm the effectiveness of our approach.

---

## Paper 84
**Title:** Enhancing Visual Continual Learning with Language-Guided Supervision  
**Venue:** CVPR 2024

**Abstract:** Continual learning in visual recognition tasks has traditionally relied on visual features alone. We propose to enhance visual continual learning by incorporating language-guided supervision. Our method leverages textual descriptions of classes to provide semantic anchors that stabilize feature representations across tasks. Language supervision helps maintain more consistent decision boundaries and reduces feature drift. The approach is compatible with various continual learning strategies and consistently improves performance when combined with existing methods. Experimental results on standard benchmarks demonstrate significant improvements in both class-incremental and task-incremental learning settings.

---

## Paper 85
**Title:** LANDER: Text-Enhanced Data-free Approach for Federated Class-Incremental Learning  
**Venue:** CVPR 2024

**Abstract:** Federated class-incremental learning (FCIL) faces the dual challenge of learning new classes continuously while preserving data privacy across distributed clients. Existing methods often require storing exemplars or generating synthetic data, raising privacy and efficiency concerns. We propose LANDER, a text-enhanced data-free approach for FCIL. LANDER leverages textual class descriptions from pre-trained language models to generate pseudo-features for rehearsal, eliminating the need to store or share any real data. The text-enhanced pseudo-features effectively approximate the distribution of real samples and enable efficient knowledge distillation. Extensive experiments demonstrate that LANDER achieves competitive performance with exemplar-based methods while maintaining strict data privacy.

---

## Paper 86
**Title:** Generative Multi-modal Models are Good Class Incremental Learners  
**Venue:** CVPR 2024

**Abstract:** Class-incremental learning (CIL) requires models to continuously learn new classes without forgetting previously learned ones. We demonstrate that generative multi-modal models can serve as effective class-incremental learners. By leveraging the generative capabilities of multi-modal models, our approach synthesizes informative training signals for previous classes, reducing the reliance on stored exemplars. The multi-modal nature of the model enables leveraging both visual and textual information for more robust incremental learning. Our method achieves an improvement of up to 14% over existing methods in few-shot CIL settings. Experiments across multiple benchmarks validate the effectiveness of generative multi-modal models for continual learning.

---

## Paper 87
**Title:** ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning  
**Venue:** CVPR 2024

**Abstract:** Continual learning for panoptic segmentation—the joint task of semantic and instance segmentation—presents unique challenges due to the complexity of dense prediction tasks. We propose ECLIPSE (Efficient Continual Learning in Panoptic Segmentation), which leverages visual prompt tuning for parameter-efficient adaptation across incremental panoptic segmentation tasks. ECLIPSE introduces task-specific visual prompts that modulate the segmentation model's behavior while keeping the backbone frozen. A prompt interaction mechanism enables knowledge sharing across tasks. Our approach significantly reduces the computational and memory overhead of continual learning for panoptic segmentation while achieving competitive or superior performance compared to full fine-tuning methods.

---

## Paper 88
**Title:** Scalable Language Model with Generalized Continual Learning  
**Venue:** ICLR 2024

**Abstract:** Continual learning has gained increasing importance as it facilitates the acquisition and refinement of scalable knowledge and skills in language models. However, existing methods typically encounter strict limitations and challenges in real-world scenarios, such as reliance on experience replay, optimization constraints, and inference task-ID. In this study, we introduce the Scalable Language Model (SLM) to overcome these limitations within a more challenging and generalized setting, representing a significant advancement toward practical applications for continual learning. Specifically, we propose the Joint Adaptive Re-Parameterization (JARe), integrated with Dynamic Task-related Knowledge Retrieval (DTKR), to enable adaptive adjustment of language models based on specific downstream tasks. This approach leverages the task distribution within the vector space, aiming to achieve a smooth and effortless continual learning process. Our method demonstrates state-of-the-art performance on diverse backbones and benchmarks, achieving effective continual learning in both full-set and few-shot scenarios with minimal forgetting. Moreover, while prior research primarily focused on a single task type such as classification, our study goes beyond, with the large language model, i.e., LLaMA-2, to explore the effects across diverse domains and task types, such that a single language model can be decently scaled to broader applications.

---

## Paper 89
**Title:** Adapting Large Language Models to Domains via Reading Comprehension  
**Venue:** ICLR 2024

**Abstract:** We explore how continued pre-training on domain-specific corpora influences large language models, revealing that training on the raw corpora endows the model with domain knowledge, but drastically hurts its prompting ability for question answering. Taken inspiration from human learning via reading comprehension—practice after reading improves the ability to answer questions based on the learned knowledge—we propose a simple method for transforming raw corpora into reading comprehension texts. Each raw text is enriched with a series of tasks related to its content. Our method, highly scalable and applicable to any pre-training corpora, consistently enhances performance across various tasks in three different domains: biomedicine, finance, and law. Notably, our 7B language model achieves competitive performance with domain-specific models of much larger scales, such as BloombergGPT-50B. Furthermore, we demonstrate that domain-specific reading comprehension texts can improve the model's performance even on general benchmarks, showing the potential to develop a general model across even more domains.

---

## Paper 90
**Title:** Dissecting Learning and Forgetting in Language Model Finetuning  
**Venue:** ICLR 2024

**Abstract:** Finetuning language models on domain-specific corpus is a common approach to enhance their domain knowledge and capability. While improving performance on domain tasks, it often brings a side-effect of forgetting of the model's general abilities. In this study, we analyze the effects of finetuning on language models by dissecting its impacts on the modeling of topic, style, and factual knowledge in text. Our method uses instruction-following LLMs such as ChatGPT to auto-generate controlled-variable text examples which we use to probe the model. Our findings reveal that finetuning results in significant shifts in the language model's topic and style priors, while actual knowledge learning only contributes to a small fraction of the total probability change. Analysis shows that the adaptation of topic and style priors behave akin to learning simple features: they are learned rapidly and require little model capacity. They are also learned independently and primarily at the beginning of a text sequence. In contrast, factual knowledge is learned stably but slowly and requires significant model capacity to learn. The research offers insights and understanding into the finer dynamics of learning and forgetting in language models, and can potentially inform future research on improving domain adaptation and addressing the challenges of forgetting in continual learning of language models.

---

## Paper 91
**Title:** TiC-CLIP: Continual Training of CLIP Models  
**Venue:** ICLR 2024

**Abstract:** Keeping large foundation models up to date on latest data is inherently expensive. To avoid the prohibitive costs of constantly retraining, it is imperative to continually train these models. This problem is exacerbated by the lack of any large scale continual learning benchmarks or baselines. We introduce the first set of web-scale Time-Continual (TiC) benchmarks for training vision-language models: TiC-DataComp, TiC-YFCC, and TiC-Redcaps. TiC-DataComp, our largest dataset, contains over 12.7B timestamped image-text pairs spanning 9 years (2014-2022). We first use our benchmarks to curate various dynamic evaluations to measure temporal robustness of existing models. We show OpenAI's CLIP (trained on data up to 2020) loses ≈8% zero-shot accuracy on our curated retrieval task from 2021-2022 compared with more recently trained models in OpenCLIP repository. We then study how to efficiently train models on time-continuous data. We demonstrate that a simple rehearsal-based approach that continues training from the last checkpoint and replays old data reduces compute by 2.5× when compared to the standard practice of retraining from scratch.

---

## Paper 92
**Title:** CPPO: Continual Learning for Reinforcement Learning with Human Feedback  
**Venue:** ICLR 2024

**Abstract:** The approach of Reinforcement Learning from Human Feedback (RLHF) is widely used for enhancing pre-trained Language Models (LM), enabling them to better align with human preferences. Existing RLHF-based LMs however require complete retraining whenever new queries or feedback are introduced, as human preferences may differ across different domains or topics. LM retraining is often impracticable in most real-world scenarios, due to the substantial time and computational costs involved, as well as data privacy concerns. To address this limitation, we propose Continual Proximal Policy Optimization (CPPO), a novel method that is able to continually align LM with dynamic human preferences. Specifically, CPPO adopts a weighting strategy to decide which samples should be utilized for enhancing policy learning and which should be used for solidifying past experiences. This seeks a good trade-off between policy learning and knowledge retention. Our experimental results show that CPPO outperforms strong Continuous learning (CL) baselines when it comes to consistently aligning with human preferences. Furthermore, compared to PPO, CPPO offers more efficient and stable learning in non-continual scenarios.

---

## Paper 93
**Title:** Learning Task-Aware Language-Image Representation for Class-Incremental Object Detection  
**Venue:** AAAI 2024

**Abstract:** Class-incremental object detection (CIOD) aims to sequentially learn new object categories while retaining the ability to detect previously learned ones. Existing methods mainly operate in the visual feature space and often suffer from catastrophic forgetting when new classes are introduced. We propose a task-aware language-image representation learning framework that leverages multi-modal features from vision-language models for CIOD. Our approach introduces task-aware prompts that encode task-specific context, enabling the model to adapt its detection capability to new classes while preserving knowledge of old classes. The integration of language supervision provides semantic regularization that stabilizes the feature representations across incremental stages. Experimental results on standard CIOD benchmarks demonstrate the effectiveness of our approach.

---

## Paper 94
**Title:** Maintaining Fairness in Logit-based Knowledge Distillation for Class-Incremental Learning  
**Venue:** AAAI 2025

**Abstract:** Knowledge distillation (KD) is a widely used technique in class-incremental learning (CIL) to mitigate catastrophic forgetting by transferring knowledge from the old model to the new one. However, we identify an inherent fairness issue in logit-based KD: the distillation process systematically favors old classes over new ones, leading to imbalanced performance. In this paper, we analyze the root cause of this fairness issue and propose a method to maintain fairness in logit-based KD for CIL. Our approach introduces a balanced distillation objective that equalizes the learning signals for old and new classes, along with a logit adjustment mechanism that accounts for the class distribution shift. Extensive experiments on benchmark datasets demonstrate that our method achieves a better balance between old and new class performance while maintaining competitive overall accuracy.

---

## Paper 95
**Title:** Sub-network Discovery and Soft-masking for Continual Learning of Mixed Tasks  
**Venue:** EMNLP 2023

**Abstract:** Continual learning in natural language processing often involves a mixture of different task types (e.g., classification, generation, extraction) arriving sequentially. We propose a method for continual learning of mixed tasks based on sub-network discovery and soft-masking. Our approach identifies task-relevant sub-networks within a shared model and applies soft masks to regulate the updates, preventing catastrophic forgetting of previously learned tasks. The soft-masking mechanism allows flexible sharing of parameters across tasks when beneficial while protecting critical parameters for specific tasks. Experiments on sequences of diverse NLP tasks demonstrate that our method effectively handles mixed-task continual learning scenarios, outperforming existing approaches that are typically designed for single-type task sequences.

---

## Paper 96
**Title:** FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning  
**Venue:** NeurIPS 2023

**Abstract:** Exemplar-free class-incremental learning (CIL) poses a significant challenge as the model must learn new classes without access to stored samples from previous tasks. Existing methods often assume homogeneous class distributions, which limits their performance in real-world scenarios where class distributions vary significantly. We propose FeCAM, which exploits the heterogeneity of class distributions by modeling each class with its own distribution in the feature space. FeCAM uses class-specific covariance matrices to capture the unique characteristics of each class distribution, enabling more accurate classification without stored exemplars. Our approach is simple, efficient, and can be easily integrated with existing continual learning methods. Extensive experiments on standard CIL benchmarks demonstrate that FeCAM achieves state-of-the-art performance in the exemplar-free setting.

---

## Paper 97
**Title:** Parameter-Level Soft-Masking for Continual Learning  
**Venue:** ICML 2023

**Abstract:** Continual learning aims to train a neural network on a sequence of tasks without catastrophic forgetting. We propose a parameter-level soft-masking approach (SPG) that learns importance scores for each parameter with respect to previously learned tasks. Unlike hard-masking methods that completely freeze parameters, our soft-masking approach allows gradual parameter updates based on their importance, enabling more flexible knowledge sharing and transfer between tasks. The soft masks are learned jointly with the task parameters and are used to modulate gradient updates during training. This approach effectively balances plasticity for new tasks and stability for old knowledge. Experiments on multiple continual learning benchmarks demonstrate that our method achieves state-of-the-art performance while being computationally efficient.

---

## Paper 98
**Title:** Continual Vision-Language Representation Learning with Off-Diagonal Information  
**Venue:** ICML 2023

**Abstract:** Continual learning of vision-language representations is essential for adapting models to evolving visual and textual data streams. Existing approaches primarily focus on preserving the diagonal entries of the contrastive similarity matrix, which represent the alignment between matched image-text pairs. However, the off-diagonal entries, which capture the relationships between unmatched pairs, also carry crucial information about the structure of the representation space. We propose Mod-X, a method that selectively aligns the off-diagonal information of contrastive matrices between the old and new models. By preserving the relative similarities among unmatched pairs, Mod-X maintains a more complete picture of the representation space structure, leading to better knowledge retention. Experiments on continual vision-language pre-training benchmarks demonstrate the effectiveness of our approach.

---

## Paper 99
**Title:** CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation  
**Venue:** ICCV 2023

**Abstract:** Vision-language pre-training has shown remarkable success in various downstream tasks. However, real-world data arrives continuously, necessitating continual pre-training of vision-language models. We propose CTP, a framework for continual vision-language pre-training that addresses the challenges of distribution shift and catastrophic forgetting. CTP introduces compatible momentum contrast to maintain consistency between current and historical representations, along with topology preservation to retain the structural relationships in the joint vision-language embedding space. We also introduce P9D, a large-scale benchmark containing over 1 million image-text pairs across diverse domains for evaluating continual vision-language pre-training. Experiments demonstrate that CTP effectively mitigates forgetting while enabling knowledge transfer to new domains.

---

## Paper 100
**Title:** Introducing Language Guidance in Prompt-based Continual Learning  
**Venue:** ICCV 2023

**Abstract:** Prompt-based continual learning has emerged as an effective approach for adapting pre-trained models to sequential tasks. However, existing prompt-based methods rely solely on visual features for prompt selection, which can lead to suboptimal task identification and knowledge retrieval. We propose LGCL (Language-Guided Continual Learning), which introduces language guidance at both the task level and class level to improve prompt-based continual learning. At the task level, language descriptions help identify the relevant task context. At the class level, semantic class information guides the learning of more discriminative prompts. The integration of language guidance enhances both the stability and plasticity of the continual learning process. Experiments on standard benchmarks demonstrate significant improvements over existing prompt-based methods.

---

## Paper 101
**Title:** Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models  
**Venue:** ICCV 2023

**Abstract:** Continual learning with vision-language models (VLMs) like CLIP offers the prospect of adapting to downstream tasks while maintaining strong zero-shot transfer capabilities. However, we show that existing continual learning methods suffer from significant zero-shot transfer degradation—the model's ability to recognize unseen classes deteriorates as it adapts to sequential tasks. We propose ZSCL (Zero-Shot Continual Learning), a method specifically designed to prevent this degradation. ZSCL introduces a reference dataset distillation strategy that efficiently preserves the zero-shot transfer capability by distilling knowledge from the original pre-trained model. We also propose the MTIL (Multi-domain Task-Incremental Learning) benchmark for evaluating zero-shot performance in continual settings. Experiments show that ZSCL achieves up to +9.7% improvement in zero-shot accuracy over existing methods while maintaining competitive incremental learning performance.

---

## Paper 102
**Title:** MRN: Multiplexed Routing Network for Incremental Multilingual Text Recognition  
**Venue:** ICCV 2023

**Abstract:** Multilingual text recognition (MTR) requires models to recognize text in multiple languages. In practice, new languages must be added incrementally, posing a continual learning challenge. We propose MRN (Multiplexed Routing Network), a novel architecture for incremental multilingual text recognition. MRN uses a multiplexed routing mechanism to dynamically select language-specific processing paths, enabling efficient adaptation to new languages without forgetting previously learned ones. The routing mechanism learns to compose shared and language-specific components, balancing parameter sharing and specialization. Experiments demonstrate that MRN achieves 10.3%-35.8% accuracy improvements over existing incremental learning approaches for multilingual text recognition.

---

## Paper 103
**Title:** Class-Incremental Grouping Network for Continual Audio-Visual Learning  
**Venue:** ICCV 2023

**Abstract:** Audio-visual learning benefits from jointly modeling audio and visual modalities. However, the continuous emergence of new audio-visual categories necessitates continual learning capabilities. We propose CIGN (Class-Incremental Grouping Network) for continual audio-visual learning. CIGN introduces a grouping mechanism that organizes audio-visual features into class-specific groups, enabling more structured and interpretable representations. The grouping structure naturally supports class-incremental learning by allowing new groups to be added for new categories while preserving existing groups for old categories. A cross-modal interaction module facilitates effective fusion of audio and visual information within each group. Extensive experiments on audio-visual classification benchmarks demonstrate the effectiveness of CIGN in the continual learning setting.

---

## Paper 104
**Title:** Continual Pre-training of Language Models  
**Venue:** ICLR 2023

**Abstract:** Language models (LMs) have been instrumental for the rapid advance of natural language processing. This paper studies continual pre-training of LMs, in particular, continual domain-adaptive pre-training (or continual DAP-training). Existing research has shown that further pre-training an LM using a domain corpus to adapt the LM to the domain can improve the end-task performance in the domain. This paper proposes a novel method to continually DAP-train an LM with a sequence of unlabeled domain corpora to adapt the LM to these domains to improve their end-task performances. The key novelty of our method is a soft-masking mechanism that directly controls the update to the LM. A novel proxy is also proposed to preserve the general knowledge in the original LM. Additionally, it contrasts the representations of the previously learned domain knowledge (including the general knowledge in the pre-trained LM) and the knowledge from the current full network to achieve knowledge integration. The method not only overcomes catastrophic forgetting, but also achieves knowledge transfer to improve end-task performances. Empirical evaluation demonstrates the effectiveness of the proposed method.

---

## Paper 105
**Title:** Progressive Prompts: Continual Learning for Language Models  
**Venue:** ICLR 2023

**Abstract:** We introduce Progressive Prompts - a simple and efficient approach for continual learning in language models. Our method allows forward transfer and resists catastrophic forgetting, without relying on data replay or a large number of task-specific parameters. Progressive Prompts learns a new soft prompt for each task and sequentially concatenates it with the previously learned prompts, while keeping the base model frozen. Experiments on standard continual learning benchmarks show that our approach outperforms state-of-the-art methods, with an improvement >20% in average test accuracy over the previous best-performing method on T5 model. We also explore a more challenging continual learning setup with longer sequences of tasks and show that Progressive Prompts significantly outperforms prior methods.

---

## Paper 106
**Title:** Class-Incremental Learning based on Label Generation  
**Venue:** ACL 2023

**Abstract:** Class-incremental learning (CIL) in natural language processing has traditionally been approached as a classification problem. We propose VAG, a novel approach that formulates CIL as a continual label generation task. Instead of learning linear classifiers that require architectural modification when new classes arrive, our method generates class labels as text sequences using a pre-trained generative language model. This formulation naturally accommodates new classes without changing the model architecture and leverages the language model's pre-trained knowledge for better generalization. We introduce techniques to mitigate forgetting in the generative setting, including constrained decoding and label-aware regularization. Experiments on text classification benchmarks demonstrate that our generation-based approach achieves competitive or superior performance compared to traditional classification-based CIL methods.

---

## Paper 107
**Title:** Analyzing and Reducing the Performance Gap in Cross-Lingual Transfer with Fine-tuning Slow and Fast  
**Venue:** ACL 2023

**Abstract:** Cross-lingual transfer, where a model trained on a source language is applied to target languages, is crucial for extending NLP capabilities to low-resource languages. However, a significant performance gap often exists between the source and target languages. In this paper, we analyze the factors contributing to this cross-lingual transfer gap and identify that catastrophic forgetting of multilingual representations during fine-tuning is a key factor. We propose a method based on differential learning rates (slow and fast) for different model components, which preserves the cross-lingual alignment learned during pre-training while adapting to the source-language task. Our approach separates the fine-tuning process into slow updates for cross-lingually important parameters and fast updates for task-specific parameters. Experiments across multiple languages and tasks demonstrate that our method significantly reduces the cross-lingual performance gap while maintaining competitive source-language performance.

---

## Paper 108
**Title:** Exploring Data Geometry for Continual Learning  
**Venue:** CVPR 2023

**Abstract:** Continual learning methods typically operate in Euclidean space, which may not adequately capture the complex geometric structures of learned representations. We explore the role of data geometry in continual learning by proposing a framework that operates in non-Euclidean geometry. Specifically, we introduce a mixed curvature space that combines hyperbolic, Euclidean, and spherical geometries to better represent hierarchical and complex relationships in the data. Our approach learns task-specific curvature parameters that adapt the geometric space to the characteristics of each task. By operating in a geometrically more appropriate space, our method achieves more stable representations across tasks and reduces forgetting. Extensive experiments on standard continual learning benchmarks demonstrate the effectiveness of incorporating data geometry into continual learning.

---

## Paper 109
**Title:** CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning  
**Venue:** CVPR 2023

**Abstract:** Computer vision models suffer from catastrophic forgetting when learning novel concepts from continuously shifting training data. Typical solutions require extensive rehearsal of previously seen data, which is impractical in many applications. We instead propose to learn a set of prompt components which are assembled with input-conditioned weights to produce input-conditioned prompts, resulting in a novel attention-based end-to-end key-query scheme. Our method, CODA-Prompt, leverages decomposed attention to dynamically compose prompts from a shared pool of components, enabling flexible adaptation to new tasks while preserving knowledge of previous ones. Our experiments show that we outperform the current SOTA method DualPrompt on established benchmarks by as much as 4.5% in average final accuracy. We also outperform the state of the art by as much as 4.4% accuracy on a continual learning benchmark which contains both class-incremental and domain-incremental task shifts.
