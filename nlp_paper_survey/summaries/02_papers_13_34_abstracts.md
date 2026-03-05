# Paper Abstracts (Papers 13-34)

---

## Paper 13: Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning
**Source:** arxiv:2507.09118 | ICCV 2025

**Abstract:**
Continual learning aims to enable models to learn sequentially from continuously incoming data while retaining performance on previously learned tasks. With the Contrastive Language-Image Pre-trained model (CLIP) exhibiting strong capabilities across various downstream tasks, there has been growing interest in leveraging CLIP for continual learning in such scenarios. Most existing works overlook the inherent modality gap in CLIP, a key factor in its generalization and adaptability. In this paper, we analyze the variations in the modality gap during the fine-tuning of vision-language pre-trained models. Our observations reveal that the modality gap effectively reflects the extent to which pre-trained knowledge is preserved. Based on these insights, we propose a simple yet effective method, MG-CLIP, that improves CLIP's performance in class-incremental learning. Our approach leverages modality gap preservation to mitigate forgetting and modality gap compensation to enhance the capacity for new data, introducing a novel modality-gap-based perspective for continual learning. Extensive experiments on multiple benchmarks demonstrate that our method outperforms existing approaches without requiring additional replay data.

---

## Paper 14: SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning
**Source:** arxiv:2411.13949

**Abstract:**
Visual instruction tuning (VIT) enables multimodal large language models (MLLMs) to effectively handle a wide range of vision tasks by framing them as language-based instructions. Building on this, continual visual instruction tuning (CVIT) extends the capability of MLLMs to incrementally learn new tasks, accommodating evolving functionalities. While prior work has advanced CVIT through the development of new benchmarks and approaches to mitigate catastrophic forgetting, these efforts largely follow traditional continual learning paradigms, neglecting the unique challenges specific to CVIT. We identify a dual form of catastrophic forgetting in CVIT, where MLLMs not only forget previously learned visual understanding but also experience a decline in instruction following abilities as they acquire new tasks. To address this, we introduce the Separable Mixture of Low-Rank Adaptation (SMoLoRA) framework, which employs separable routing through two distinct modules—one for visual understanding and another for instruction following. This dual-routing design enables specialized adaptation in both domains, preventing forgetting while improving performance. Furthermore, we propose a new CVIT benchmark that goes beyond existing benchmarks by additionally evaluating a model's ability to generalize to unseen tasks and handle diverse instructions across various tasks. Extensive experiments demonstrate that SMoLoRA outperforms existing methods in mitigating dual forgetting, improving generalization to unseen tasks, and ensuring robustness in following diverse instructions.

---

## Paper 15: DMNSP: Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning
**Source:** ICCV 2025 (CVF Open Access)

**Abstract:**
Vision-Language Models (VLM) have emerged as a highly promising approach for Continual Learning (CL) due to their powerful generalized features. While adapter-based VLM can exploit both task-specific and task-agnostic features, current CL methods have largely overlooked the distinct and evolving parameter distributions in visual and language modalities, which are found crucial for effectively mitigating catastrophic forgetting. In this study, we find that the visual modality experiences a broader parameter distribution and propose DMNSP (Dynamic Multi-Layer Null Space Projection), which dynamically projects updates into the null space of previous tasks across multiple layers, separately handling the visual and language modalities to preserve learned knowledge while accommodating new tasks.

*(Note: Full abstract obtained from Google Scholar snippet + CVF proceedings. Paper is ICCV 2025 open-access only, not on arxiv.)*

---

## Paper 16: Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering
**Source:** arxiv:2502.04469 | ICCV 2025

**Abstract:**
Continual Learning in Visual Question Answering (VQACL) requires models to acquire new visual-linguistic skills (plasticity) while preserving previously learned knowledge (stability). The inherent multimodality of VQACL exacerbates this challenge, as models must balance stability across visual and textual domains while adapting to novel objects and reasoning tasks. Existing methods, primarily designed for unimodal settings, often fall short in addressing this dual requirement. In this work, we present QUestion-only replay with Attention Distillation (QUAD), a novel approach for VQACL that leverages only past task questions for regularization. By eliminating the need to store visual data, QUAD not only reduces memory overhead, but also alleviates privacy concerns. Our method introduces a Question-only Replay mechanism that selectively reuses prior task questions to counteract overfitting to the answer space of the current task, addressing the problem out of answer set. Complementing this, we propose Attention Consistency Distillation to enforce both intra-modal and inter-modal attention consistency across tasks, preserving essential visual-linguistic associations. Extensive experiments on VQAv2 and NExT-QA demonstrate that QUAD significantly outperforms state-of-the-art methods, achieving robust performance in continual VQA.

---

## Paper 17: TWIST & SCOUT: Grounding Multimodal LLM-Experts by Forget-Free Tuning
**Source:** arxiv:2410.10491

**Abstract:**
Spatial awareness is key to enable embodied multimodal AI systems. Yet, without vast amounts of spatial supervision, current Multimodal Large Language Models (MLLMs) struggle at this task. In this paper, we introduce TWIST & SCOUT, a framework that equips pre-trained MLLMs with visual grounding ability without forgetting their existing image and language understanding skills. To this end, we propose TWIST, a twin-expert stepwise tuning module that modifies the decoder of the language model using one frozen module pre-trained on image understanding tasks and another learnable one for visual grounding tasks. This allows the MLLM to retain previously learned knowledge and skills, while acquiring what is missing. To fine-tune the model effectively, we generate a high-quality synthetic dataset we call SCOUT, which mimics human reasoning in visual grounding. This dataset provides rich supervision signals, describing a step-by-step multimodal reasoning process, thereby simplifying the task of visual grounding. We evaluate our approach on several standard benchmark datasets, encompassing grounded image captioning, zero-shot localization, and visual grounding tasks. Our method consistently delivers strong performance across all tasks, while retaining the pre-trained image understanding capabilities.

---

## Paper 18: Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models
**Source:** arxiv:2508.00260 | ICCV 2025

**Abstract:**
Continual learning enables pre-trained generative vision-language models (VLMs) to incorporate knowledge from new tasks without retraining data from previous ones. Recent methods update a visual projector to translate visual information for new tasks, connecting pre-trained vision encoders with large language models. However, such adjustments may cause the models to prioritize visual inputs over language instructions, particularly learning tasks with repetitive types of textual instructions. To address the neglect of language instructions, we propose a novel framework that grounds the translation of visual information on instructions for language models. We introduce a mixture of visual projectors, each serving as a specialized visual-to-language translation expert based on the given instruction context to adapt to new tasks. To avoid using experts for irrelevant instruction contexts, we propose an expert recommendation strategy that reuses experts for tasks similar to those previously learned. Additionally, we introduce expert pruning to alleviate interference from the use of experts that cumulatively activated in previous tasks. Extensive experiments on diverse vision-language tasks demonstrate that our method outperforms existing continual learning approaches by generating instruction-following responses.

---

## Paper 19: External Knowledge Injection for CLIP-Based Class-Incremental Learning
**Source:** arxiv:2503.08510 | ICCV 2025

**Abstract:**
Class-Incremental Learning (CIL) enables learning systems to continuously adapt to evolving data streams. With the advancement of pre-training, leveraging pre-trained vision-language models (e.g., CLIP) offers a promising starting point for CIL. However, CLIP makes decisions by matching visual embeddings to class names, overlooking the rich contextual information conveyed through language. For instance, the concept of "cat" can be decomposed into features like tail, fur, and face for recognition. Besides, since the model is continually updated, these detailed features are overwritten in CIL, requiring external knowledge for compensation. In this paper, we introduce ExterNal knowledGe INjEction (ENGINE) for CLIP-based CIL. To enhance knowledge transfer from outside the dataset, we propose a dual-branch injection tuning framework that encodes informative knowledge from both visual and textual modalities. The visual branch is enhanced with data augmentation to enrich the visual features, while the textual branch leverages GPT-4 to rewrite discriminative descriptors. In addition to this on-the-fly knowledge injection, we also implement post-tuning knowledge by re-ranking the prediction results during inference. With the injected knowledge, the model can better capture informative features for downstream tasks as data evolves. Extensive experiments demonstrate the state-of-the-art performance of ENGINE.

---

## Paper 20: Overcoming Dual Drift for Continual Long-Tailed Visual Question Answering
**Source:** ICCV 2025 (CVF Open Access)

**Abstract:**
Visual Question Answering (VQA) is a widely explored multimodal task aimed at answering questions based on images. Recently, a few studies have started to investigate continual learning in VQA to cope with evolving multimodal data streams. However, these studies fall short of tackling another critical issue in real-world VQA applications: the long-tailed distribution of data. In this paper, we introduce Continual Long-Tailed Visual Question Answering (CLT-VQA) and identify two critical challenges: inner-task prototype drift, where class prototypes shift due to long-tailed imbalance within each task, and inter-task prototype drift, where prototypes of old tasks shift when learning new tasks. To overcome these dual drifts, the authors propose methods to stabilize prototypes and maintain balanced representations across the evolving task sequence.

*(Note: Full abstract obtained from Google Scholar snippet + CVF proceedings. Paper is ICCV 2025 open-access only, not on arxiv.)*

---

## Paper 21: PLAN: Proactive Low-Rank Allocation for Continual Learning
**Source:** arxiv:2510.21188 | ICCV 2025

**Abstract:**
Continual learning (CL) requires models to continuously adapt to new tasks without forgetting past knowledge. In this work, we propose Proactive Low-rank AllocatioN (PLAN), a framework that extends Low-Rank Adaptation (LoRA) to enable efficient and interference-aware fine-tuning of large pre-trained models in CL settings. PLAN proactively manages the allocation of task-specific subspaces by introducing orthogonal basis vectors for each task and optimizing them through a perturbation-based strategy that minimizes conflicts with previously learned parameters. Furthermore, PLAN incorporates a novel selection mechanism that identifies and assigns basis vectors with minimal sensitivity to interference, reducing the risk of degrading past knowledge while maintaining efficient adaptation to new tasks. Empirical results on standard CL benchmarks demonstrate that PLAN consistently outperforms existing methods, establishing a new state-of-the-art for continual learning with foundation models.

---

## Paper 22: Knowledge Decoupling via Orthogonal Projection for Lifelong Editing of Large Language Models
**Source:** ACL 2025 (aclanthology.org/2025.acl-long.646/)

**Abstract:**
As large language models (LLMs) require continuous knowledge updates and the mitigation of hallucination issues in generated content, lifelong model editing has become a prominent research area. A mainstream knowledge editing method usually freezes LLM's original parameters and adds extra trainable modules for new knowledge management, reducing interference with old knowledge. Although these approaches have achieved some success, our experiments show that, after extensive editing, the model's knowledge understanding and memory capacity significantly degrade, particularly concerning early edited knowledge. The root cause is that subsequent edits interfere with the previously edited knowledge, and we refer to this phenomenon as knowledge coupling. To address this issue, we propose the Knowledge Decoupling Editing (KDE) method. Specifically, KDE stores the basis vectors of the representation space of past edits in a knowledge cache. It projects the gradient of the current edit onto a space orthogonal to previous knowledge for updating. This method effectively alleviates the coupling between different pieces of knowledge. We also propose a two-stage training strategy to better balance the model's ability to edit new knowledge and distinguish whether a query is related to previous edits. This strategy gradually reduces the interference between new knowledge editing and query distinction, maintaining stable performance during long-term editing. We compared KDE with nine cutting-edge editing methods across multiple mainstream LLMs. The results demonstrate that, regarding question-answering ability and hallucination mitigation, KDE achieves average improvements of 14% and 61%.

---

## Paper 23: Serial Lifelong Editing via Mixture of Knowledge Experts
**Source:** ACL 2025 (aclanthology.org/2025.acl-long.1492/)

**Abstract:**
It is challenging to update Large language models (LLMs) since real-world knowledge evolves. While existing Lifelong Knowledge Editing (LKE) methods efficiently update sequentially incoming edits, they often struggle to precisely overwrite the outdated knowledge with the latest one, resulting in conflicts that hinder LLMs from determining the correct answer. To address this Serial Lifelong Knowledge Editing (sLKE) problem, we propose a novel Mixture-of-Knowledge-Experts scheme with an Activation-guided Routing Mechanism (ARM), which assigns specialized experts to store domain-specific knowledge and ensures that each update completely overwrites old information with the latest data. Furthermore, we introduce a novel sLKE benchmark where answers to the same concept are updated repeatedly, to assess the ability of editing methods to refresh knowledge accurately. Experimental results on both LKE and sLKE benchmarks show that our ARM performs favorably against SOTA knowledge editing methods.

---

## Paper 24: Efficient Domain Continual Pretraining by Mitigating the Stability Gap
**Source:** ACL 2025 (aclanthology.org/2025.acl-long.1578/)

**Abstract:**
Continual pretraining enables Large Language Models (LLMs) to adapt to specialized domains like medicine and law. However, we observe a consistent phenomenon across different model sizes and domains: a temporary performance drop at the start of the continual pretraining process, followed by a performance recovery phase. To gain a deeper understanding of this issue, we use the stability gap—a concept adapted from the visual domain—which explains this initial drop arises from instability in the model's general abilities. We validate this hypothesis through a series of experiments. To address this initial instability and enhance LLM performance within a fixed compute budget, we propose a training strategy that mitigates instability by increasing the number of epochs, alongside two data sampling strategies targeting data domain relevance and corpus distribution. We conduct experiments on Llama-family models to validate the effectiveness of our strategies for continual pretraining and instruction tuning in medical and legal domains. Our strategies improve the average medical task performance of the OpenLlama-3B model from 36.2% to 40.7% using only 40% of the original training budget, while also enhancing general task performance without causing forgetting. Furthermore, we apply our strategies to continually pre-train and instruction-tune the Llama-3-8B model. The resulting model, Llama-3-Physician, achieves the best medical performance among open-source models on several benchmarks and rivals GPT-4 on specific tasks.

---

## Paper 25: NSE: Neuron-Level Sequential Editing for Large Language Models
**Source:** arxiv:2410.04045 | ACL 2025

**Abstract:**
This work explores sequential model editing in large language models (LLMs), a critical task that involves modifying internal knowledge within LLMs continuously through multi-round editing, each incorporating updates or corrections to adjust the model outputs without the need for costly retraining. Existing model editing methods, especially those that alter model parameters, typically focus on single-round editing and often face significant challenges in sequential model editing—most notably issues of model forgetting and failure. To address these challenges, we introduce a new model editing method, namely Neuron-level Sequential Editing (NSE), tailored for supporting sequential model editing. Specifically, we optimize the target layer's hidden states using the model's original weights to prevent model failure. Furthermore, we iteratively select neurons in multiple layers for editing based on their activation values to mitigate model forgetting. Our empirical experiments demonstrate that NSE significantly outperforms current modifying parameters model editing methods, marking a substantial advancement in the field of sequential model editing.

---

## Paper 26: CLoRA: Controlled Low-Rank Adaptation with Subspace Regularization
**Source:** arxiv:2410.16801 | ACL 2025

**Abstract:**
Large language models (LLMs) exhibit remarkable capabilities in natural language processing but face catastrophic forgetting when learning new tasks, where adaptation to a new domain leads to a substantial decline in performance on previous tasks. In this paper, we propose Controlled LoRA (CLoRA), a sub-space regularization method on LoRA structure. Aiming to reduce the scale of output change while introducing minimal constraint on model capacity, CLoRA imposes constraint on the direction of updating matrix's null space. Experimental results on one-stage LLM finetuning tasks and continual learning settings highlight the superiority of CLoRA as an effective parameter efficient finetuning method with catastrophic forgetting mitigating. Further investigation for model parameters indicates that CLoRA effectively balances the trade-off between model capacity and degree of forgetting.

---

## Paper 27: HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model
**Source:** arxiv:2503.12941 | ACL 2025 (Main)

**Abstract:**
Instruction tuning is widely used to improve a pre-trained Multimodal Large Language Model (MLLM) by training it on curated task-specific datasets, enabling better comprehension of human instructions. However, it is infeasible to collect all possible instruction datasets simultaneously in real-world scenarios. Thus, enabling MLLM with continual instruction tuning is essential for maintaining their adaptability. However, existing methods often trade off memory efficiency for performance gains, significantly compromising overall efficiency. In this paper, we propose a task-specific expansion and task-general fusion framework based on the variations in Centered Kernel Alignment (CKA) similarity across different model layers when trained on diverse datasets. Furthermore, we analyze the information leakage present in the existing benchmark and propose a new and more challenging benchmark to rationally evaluate the performance of different methods. Comprehensive experiments showcase a significant performance improvement of our method compared to existing state-of-the-art methods.

---

## Paper 28: Multi-Modality Expansion and Retention for LLMs through Parameter Merging and Decoupling
**Source:** arxiv:2505.17110 | ACL 2025

**Abstract:**
Fine-tuning Large Language Models (LLMs) with multimodal encoders on modality-specific data expands the modalities that LLMs can handle, leading to the formation of Multimodal LLMs (MLLMs). However, this paradigm heavily relies on resource-intensive and inflexible fine-tuning from scratch with new multimodal data. In this paper, we propose MMER (Multi-modality Expansion and Retention), a training-free approach that integrates existing MLLMs for effective multimodal expansion while retaining their original performance. Specifically, MMER reuses MLLMs' multimodal encoders while merging their LLM parameters. By comparing original and merged LLM parameters, MMER generates binary masks to approximately separate LLM parameters for each modality. These decoupled parameters can independently process modality-specific inputs, reducing parameter conflicts and preserving original MLLMs' fidelity. MMER can also mitigate catastrophic forgetting by applying a similar process to MLLMs fine-tuned on new tasks. Extensive experiments show significant improvements over baselines, proving that MMER effectively expands LLMs' multimodal capabilities while retaining 99% of the original performance, and also markedly mitigates catastrophic forgetting.

---

## Paper 29: GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
**Source:** arxiv:2507.02503 | ACL 2025 (Main)

**Abstract:**
Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches.

---

## Paper 30: DGAR: A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning
**Source:** arxiv:2506.04083 | ACL 2025

**Abstract:**
Recent Continual Learning (CL)-based Temporal Knowledge Graph Reasoning (TKGR) methods focus on significantly reducing computational cost and mitigating catastrophic forgetting caused by fine-tuning models with new data. However, existing CL-based TKGR methods still face two key limitations: (1) They usually one-sidedly reorganize individual historical facts, while overlooking the historical context essential for accurately understanding the historical semantics of these facts; (2) They preserve historical knowledge by simply replaying historical facts, while ignoring the potential conflicts between historical and emerging facts. In this paper, we propose a Deep Generative Adaptive Replay (DGAR) method, which can generate and adaptively replay historical entity distribution representations from the whole historical context. To address the first challenge, historical context prompts as sampling units are built to preserve the whole historical context information. To overcome the second challenge, a pre-trained diffusion model is adopted to generate the historical distribution. During the generation process, the common features between the historical and current distributions are enhanced under the guidance of the TKGR model. In addition, a layer-by-layer adaptive replay mechanism is designed to effectively integrate historical and current distributions. Experimental results demonstrate that DGAR significantly outperforms baselines in reasoning and mitigating forgetting.

---

## Paper 31: Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory
**Source:** ACL 2025 (aclanthology.org/2025.acl-long.1385/)

**Abstract:**
Semiparametric language models (LMs) have shown promise in various Natural Language Processing (NLP) tasks. However, they utilize non-parametric memory as static storage, which lacks learning capability and remains disconnected from the internal information flow of the parametric models, limiting scalability and efficiency. Based on recent interpretability theories of LMs, we reconceptualize the non-parametric memory represented by kNN-LM as a learnable Mixture-of-Neighbors Induction Memory (MoNIM), which synergizes the induction capabilities of attention heads with the memorization strength of feed-forward networks (FFN). By integrating into the model's information flow, MoNIM functions as an FFN-like bypass layer within the Transformer architecture, enabling effective learning of new knowledge. Extensive experiments demonstrate that MoNIM is a retentive and scalable continual learner in both data- and model-wise, enhancing the scalability and continual learning performance of semiparametric LMs.

---

## Paper 32: Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning
**Source:** arxiv:2403.10056 | ACL 2025

**Abstract:**
Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score, to measure the generalization and instruction-following abilities of LLMs. Experiments demonstrate our method achieves superior performance on both seen and held-out tasks.

---

## Paper 33: Recurrent Knowledge Identification and Fusion for Language Model Continual Learning
**Source:** arxiv:2502.17510 | ACL 2025 (Main)

**Abstract:**
Continual learning (CL) is crucial for deploying large language models (LLMs) in dynamic real-world environments without costly retraining. While recent model ensemble and model merging methods guided by parameter importance have gained popularity, they often struggle to balance knowledge transfer and forgetting, mainly due to the reliance on static importance estimates during sequential training. In this paper, we present Recurrent-KIF, a novel CL framework for Recurrent Knowledge Identification and Fusion, which enables dynamic estimation of parameter importance distributions to enhance knowledge transfer. Inspired by human continual learning, Recurrent-KIF employs an inner loop that rapidly adapts to new tasks while identifying important parameters, coupled with an outer loop that globally manages the fusion of new and historical knowledge through redundant knowledge pruning and key knowledge merging. These inner-outer loops iteratively perform multiple rounds of fusion, allowing Recurrent-KIF to leverage intermediate training information and adaptively adjust fusion strategies based on evolving importance distributions. Extensive experiments on two CL benchmarks with various model sizes (from 770M to 13B) demonstrate that Recurrent-KIF effectively mitigates catastrophic forgetting and enhances knowledge transfer.

---

## Paper 34: TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining
**Source:** arxiv:2504.02107 | ACL 2025

**Abstract:**
Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) — orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains.
