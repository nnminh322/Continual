# Paper Abstracts: Papers 35–61

## ICML 2025 (Papers 35–48)

---

### Paper 35
**Title:** Exploiting Presentative Feature Distributions for Parameter-Efficient Continual Learning of Large Language Models
**Venue:** ICML 2025 | **Category:** [LLM-CL]
**Source:** OpenReview 6udKBHc0Mr

**Abstract:**
Endowing large language models (LLMs) with continual learning (CL) capacities has attracted increasing attention. However, LLMs typically entail substantial computational costs, and their deployment in CL scenarios exacerbates the problem. To address this, Parameter-Efficient Fine-Tuning (PEFT) methods have been widely adopted. Nevertheless, existing PEFT-based CL approaches often rely on expanding architectures or identifying task-specific components, which either increase model complexity or demand additional task information. In this work, we propose a novel CL method that characterizes each PEFT block by its presentative feature distribution—a compact statistical representation capturing the knowledge encoded in the block. When confronted with new data, our method dynamically selects the most appropriate PEFT block based on distribution similarity and determines whether to reuse an existing block or create a new one. This strategy enables efficient knowledge sharing and reduces redundant parameterization across tasks. Extensive experiments on multiple CL benchmarks demonstrate that our approach achieves competitive or superior performance compared to state-of-the-art methods while maintaining parameter efficiency.

---

### Paper 36
**Title:** Reinforced Lifelong Editing for Language Models
**Venue:** ICML 2025 | **Category:** [KE]
**Source:** arxiv:2502.05759

**Abstract:**
Knowledge editing enables efficient modification of language models' behaviors without full retraining. However, existing methods degrade as edits accumulate over time, failing in lifelong editing scenarios. We propose RLEdit, a reinforcement learning-based editing method that treats editing losses as rewards and learns an adaptive editing policy. RLEdit introduces a lightweight hypernetwork trained via policy gradient methods to generate context-aware parameter updates. Our approach naturally handles sequential edits by learning from the history of previous modifications, maintaining a balance between accommodating new edits and preserving existing knowledge. Experiments across multiple benchmarks show RLEdit achieves 59.24% improvement in lifelong editing performance while requiring only 2.11% of the time compared to leading baselines.

---

### Paper 37
**Title:** WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs
**Venue:** ICML 2025 | **Category:** [KE]
**Source:** arxiv:2503.05683

**Abstract:**
Knowledge editing methods promise to update the knowledge of large language models efficiently without expensive retraining. However, their effectiveness in realistic lifelong editing settings remains poorly understood. We introduce WikiBigEdit, a large-scale benchmark derived from real-world Wikidata edits containing over 500K question-answer pairs. Using this benchmark, we conduct a comprehensive evaluation of state-of-the-art knowledge editing methods in truly lifelong settings with thousands of sequential edits. Our findings reveal significant limitations: all methods show substantial performance degradation as edits accumulate, with reliability dropping and unintended side effects increasing. We analyze the root causes of these failures and identify key factors including edit interference, parameter saturation, and knowledge propagation failures. Our work provides critical insights for the development of more robust lifelong editing approaches.

---

### Paper 38
**Title:** Knowledge Swapping via Learning and Unlearning
**Venue:** ICML 2025 | **Category:** [KE]
**Source:** arxiv:2502.08075

**Abstract:**
We introduce the task of Knowledge Swapping, which aims to simultaneously inject new knowledge and remove outdated or undesired knowledge from language models. Unlike traditional knowledge editing that only adds or modifies knowledge, or machine unlearning that only removes knowledge, Knowledge Swapping requires both operations to be performed coherently. We propose a "Learning Before Forgetting" strategy that first injects the new replacement knowledge before unlearning the outdated knowledge, which we find is more effective than the reverse order. Our approach uses a two-stage pipeline with constrained optimization to ensure the new knowledge is robustly acquired while the old knowledge is thoroughly removed. Experiments demonstrate that our method achieves effective knowledge swapping across multiple domains while maintaining model utility on unrelated tasks.

---

### Paper 39
**Title:** Learning Dynamics in Continual Pre-Training for Large Language Models
**Venue:** ICML 2025 (Oral) | **Category:** [Analysis]
**Source:** arxiv:2505.07796

**Abstract:**
Continual pre-training (CPT) adapts pre-trained language models to new domains or corpora through additional pre-training. Despite its practical importance, the learning dynamics underlying CPT remain poorly understood. In this work, we present a comprehensive empirical and theoretical analysis of CPT dynamics. We identify two critical phenomena: the "distribution shift effect," where differences between pre-training and new data distributions lead to initial performance degradation, and the "learning rate annealing effect," where the reduced learning rate schedule in CPT limits adaptation speed. Building on these insights, we derive a CPT scaling law that unifies both effects, enabling practitioners to predict CPT outcomes based on data distribution divergence and training hyperparameters. Our scaling law accurately predicts CPT performance across diverse settings and provides practical guidelines for optimizing CPT efficiency.

---

### Paper 40
**Title:** Large Continual Instruction Assistant
**Venue:** ICML 2025 | **Category:** [IT]
**Source:** arxiv:2410.10868

**Abstract:**
We propose a general Continual Instruction Tuning (CIT) framework for large language models that enables continuous learning across diverse instruction-following tasks. Our approach introduces an Exponential Moving Average (EMA) update mechanism with a novel plasticity-stability balanced coefficient that dynamically adjusts the trade-off between learning new tasks and preserving previously acquired capabilities. The framework maintains a small set of representative exemplars from previous tasks and uses them in conjunction with the EMA update to prevent catastrophic forgetting. We demonstrate that our method effectively handles continuous streams of instruction tuning data across varied task types including generation, summarization, question answering, and reasoning, achieving strong performance on both new and previously learned tasks.

---

### Paper 41
**Title:** TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by Hierarchical Gradient-Similarity Tree
**Venue:** ICML 2025 | **Category:** [LoRA]
**Source:** arxiv:2506.10355

**Abstract:**
Continual learning with Low-Rank Adaptation (LoRA) has shown promise for efficiently adapting large pre-trained models across sequential tasks. However, existing approaches typically apply uniform adaptation across all model layers, ignoring the fact that different layers contribute differently to different tasks. We propose TreeLoRA, a method that assigns layer-wise LoRA adapters based on a hierarchical gradient similarity tree. Our approach first computes gradient-based task similarity at each layer, then organizes this information into a hierarchical tree structure that guides adapter sharing and allocation decisions. Using multi-armed bandit techniques, TreeLoRA dynamically decides at each layer whether to reuse an existing adapter, share adapters across tasks, or create a new one. This fine-grained, layer-level control significantly reduces parameter overhead while improving knowledge transfer between related tasks. Extensive experiments across multiple continual learning benchmarks demonstrate that TreeLoRA achieves superior performance with fewer parameters compared to existing LoRA-based continual learning methods.

---

### Paper 42
**Title:** ALKN: Adaptive Localization of Knowledge Negation for Continual LLM Unlearning
**Venue:** ICML 2025 | **Category:** [KE]
**Source:** OpenReview tcK4PV3VN4

**Abstract:**
Machine unlearning for large language models (LLMs) aims to remove specific undesired knowledge while maintaining model utility on other tasks. In continual unlearning scenarios where multiple unlearning requests arrive sequentially, existing methods suffer from accumulated utility degradation. We propose ALKN (Adaptive Localization of Knowledge Negation), which introduces a dynamic masking mechanism to sparsify training gradients during unlearning, focusing modifications on the most relevant model parameters. ALKN adaptively adjusts the unlearning intensity based on the relationship between the target knowledge and the model's existing knowledge structure. By localizing the gradient updates to a small subset of critical parameters, our method minimizes collateral damage to retained knowledge. Experiments on sequential unlearning benchmarks show that ALKN effectively removes target knowledge while preserving significantly more model utility compared to existing approaches, especially when facing a long sequence of continual unlearning requests.

---

### Paper 43
**Title:** From RAG to Memory: Non-Parametric Continual Learning for Large Language Models
**Venue:** ICML 2025 | **Category:** [LLM-CL]
**Source:** arxiv:2502.14802

**Abstract:**
Retrieval-Augmented Generation (RAG) enables language models to access external knowledge without parameter updates, but current RAG systems lack the ability to integrate, consolidate, and update retrieved information over time—capabilities essential for true continual learning. We present HippoRAG 2, a framework that bridges the gap between RAG and human-like long-term memory by introducing non-parametric continual learning mechanisms. Our approach augments RAG with a knowledge graph-based memory that can automatically integrate new information, resolve conflicts with existing knowledge, and form associative connections between related concepts. Experiments demonstrate a 7% improvement on associative memory tasks and strong performance on continual knowledge integration benchmarks, showing that non-parametric approaches offer a promising path toward continual learning for LLMs without the risks of catastrophic forgetting inherent in parametric updates.

---

### Paper 44
**Title:** SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning
**Venue:** ICML 2025 | **Category:** [MM]
**Source:** arxiv:2505.02486

**Abstract:**
Multimodal large language models (MLLMs) face catastrophic forgetting during continual instruction tuning across diverse multimodal tasks. We identify and categorize forgetting in this setting into two distinct types: superficial forgetting, where the model's output format deviates from expected patterns while retaining underlying knowledge, and essential forgetting, where the model genuinely loses previously acquired knowledge. To address these complementary challenges, we propose SEFE (Superficial and Essential Forgetting Eliminator). For superficial forgetting, we introduce Answer Style Diversification, a data augmentation strategy that exposes the model to varied answer formats during training, making it more robust to format shifts. For essential forgetting, we propose RegLoRA, a regularized Low-Rank Adaptation method that constrains parameter updates to preserve critical knowledge while maintaining plasticity for learning new tasks. Experiments on multiple multimodal continual learning benchmarks demonstrate that SEFE effectively mitigates both types of forgetting, achieving state-of-the-art performance.

---

### Paper 45
**Title:** LADA: Scalable Label-Specific CLIP Adapter for Continual Learning
**Venue:** ICML 2025 | **Category:** [VL]
**Source:** arxiv:2505.23271

**Abstract:**
Continual learning with pre-trained vision-language models like CLIP faces the challenge of adapting to sequential tasks without forgetting. Existing adapter-based methods often ignore class-level information, leading to interference between classes across different tasks. We propose LADA (Label-specific Adapter), a scalable continual learning framework that appends lightweight label-specific memory units to a frozen CLIP backbone. Each memory unit captures class-specific features and is updated only when learning its corresponding class, naturally preventing interference with other classes. We further introduce a feature distillation mechanism that aligns the adapted features with the original CLIP feature space, preserving the model's zero-shot generalization capabilities. LADA requires minimal additional parameters per class and supports efficient inference without task identity. Experiments on standard continual learning benchmarks demonstrate that LADA achieves state-of-the-art performance while maintaining the scalability and generalization benefits of CLIP.

---

### Paper 46
**Title:** Componential Prompt-Knowledge Alignment for Domain Incremental Learning
**Venue:** ICML 2025 | **Category:** [VL]
**Source:** arxiv:2505.04575

**Abstract:**
Prompt-based methods have shown promise for continual learning with pre-trained vision-language models, but they often suffer from misalignment between domain-specific prompts and the model's internal knowledge representations. We identify a fundamental issue we term component-wise misalignment: different components of learned prompts (e.g., those responsible for feature extraction vs. classification) may be specialized to different previous domains rather than the current one. We propose KA-Prompt (Knowledge-Aligned Prompt), a framework that explicitly addresses this misalignment by decomposing prompts into functional components and ensuring each component is properly aligned with domain-specific knowledge. Our approach introduces a componential alignment loss that encourages consistent domain specialization across prompt components, along with a knowledge-guided prompt selection mechanism for inference. Experiments on domain incremental learning benchmarks demonstrate that KA-Prompt significantly outperforms existing prompt-based methods by resolving the component-wise misalignment problem.

---

### Paper 47
**Title:** Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting
**Venue:** ICML 2025 | **Category:** [VL]
**Source:** arxiv:2505.24088

**Abstract:**
Fine-tuning vision foundation models (VFMs) on downstream tasks often leads to catastrophic forgetting of pre-trained knowledge. Existing methods attempt to preserve feature distributions but rely on storing exemplars or computing expensive statistics over entire datasets. We propose Proxy-FDA (Proxy-based Feature Distribution Alignment), which uses nearest neighbor graphs to construct informative proxies that compactly represent the feature distribution of pre-trained models. During fine-tuning, our method aligns the evolving feature distribution with these proxies, effectively preserving the structure of the original feature space while allowing adaptation to new tasks. The proxy-based approach is memory-efficient and computationally lightweight, requiring no stored exemplars from previous tasks. Experiments across multiple continual learning and domain adaptation benchmarks demonstrate that Proxy-FDA effectively prevents forgetting while achieving competitive fine-tuning performance on downstream tasks.

---

### Paper 48
**Title:** Understanding the Forgetting of Replay-based Continual Learning via Feature Learning: Angle Matters
**Venue:** ICML 2025 | **Category:** [Analysis]
**Source:** OpenReview 6UIer20oYA

**Abstract:**
Replay-based methods are among the most effective approaches for continual learning, yet a theoretical understanding of how and why they mitigate forgetting remains limited. We develop a unified theoretical framework for analyzing replay-based continual learning through the lens of feature learning. Our key insight is that the angle between task signal vectors plays a crucial role in determining the degree of forgetting. When task signals are more aligned (smaller angle), replay is more effective at preventing forgetting; when they are more orthogonal, forgetting becomes harder to mitigate. We formalize this insight through a feature learning theory that characterizes how replay influences the learned representations across sequential tasks. Our analysis reveals that replay effectiveness depends not just on the number of stored exemplars but fundamentally on the geometric relationship between task-specific features in the representation space. Experiments validate our theoretical predictions across multiple continual learning settings and architectures.

---

## ICLR 2025 (Papers 49–56)

---

### Paper 49
**Title:** LOIRE: LifelOng learning on Incremental data via pre-trained LM gRowth Efficiently
**Venue:** ICLR 2025 | **Category:** [LLM-CL]
**Source:** OpenReview F5PlYMC5ik

**Abstract:**
Pre-trained language models (PLMs) often require continual learning to stay current with evolving knowledge and domains. However, existing approaches either fine-tune the entire model (risking forgetting) or freeze it (limiting adaptation). We propose LOIRE, a framework where PLMs grow their capacity using incremental data through a novel plug-in layer growth operator. Instead of modifying existing parameters, LOIRE adds new lightweight layers that are trained on incoming data while preserving the original model's knowledge. The growth operator determines when and where to add capacity based on the novelty of incoming data. LOIRE reduces computational expenses by 29.22% compared to full fine-tuning while achieving competitive or superior performance on continual learning benchmarks across multiple domains and languages.

---

### Paper 50
**Title:** On Large Language Model Continual Unlearning
**Venue:** ICLR 2025 | **Category:** [KE]
**Source:** arxiv:2407.10223

**Abstract:**
While large language models have demonstrated impressive performance across various domains and tasks, their security issues have become increasingly severe. Machine unlearning has emerged as a representative approach for model safety and security by removing the influence of undesired data on the target model. However, these methods do not sufficiently consider that unlearning requests in real-world scenarios are continuously emerging, especially in the context of LLMs, which may lead to accumulated model utility loss that eventually becomes unacceptable. Moreover, existing LLM unlearning methods often ignore previous data access limitations due to privacy concerns and copyright protection. Without previous data, the utility preservation during unlearning is much harder. To overcome these challenges, we propose the OOO framework that includes an Orthogonal low-rank adapter (LoRA) for continually unlearning requested data and an Out-Of-Distribution (OOD) detector to measure the similarity between input and unlearning data. The orthogonal LoRA achieves parameter disentanglement among continual unlearning requests. The OOD detector is trained with a novel contrastive entropy loss and utilizes a glocal-aware scoring mechanism. During inference, our OOO framework can decide whether and to what extent to load the unlearning LoRA based on the OOD detector's predicted similarity between the input and the unlearned knowledge. Notably, OOO's effectiveness does not rely on any retained data. We conducted extensive experiments on OOO and state-of-the-art LLM unlearning methods across three tasks and seven datasets. The results indicate that OOO consistently achieves the best unlearning effectiveness and utility preservation, especially when facing continuous unlearning requests.

---

### Paper 51
**Title:** SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning
**Venue:** ICLR 2025 (Oral) | **Category:** [LoRA]
**Source:** arxiv:2501.13198

**Abstract:**
Continual Learning (CL) with foundation models has recently emerged as a promising paradigm to exploit abundant knowledge acquired during pre-training for tackling sequential tasks. However, existing prompt-based and Low-Rank Adaptation-based (LoRA-based) methods often require expanding a prompt/LoRA pool or retaining samples of previous tasks, which poses significant scalability challenges as the number of tasks grows. To address these limitations, we propose Scalable Decoupled LoRA (SD-LoRA) for class incremental learning, which continually separates the learning of the magnitude and direction of LoRA components without rehearsal. Our empirical and theoretical analysis reveals that SD-LoRA tends to follow a low-loss trajectory and converges to an overlapping low-loss region for all learned tasks, resulting in an excellent stability-plasticity trade-off. Building upon these insights, we introduce two variants of SD-LoRA with further improved parameter efficiency. All parameters of SD-LoRAs can be end-to-end optimized for CL objectives. Meanwhile, they support efficient inference by allowing direct evaluation with the finally trained model, obviating the need for component selection. Extensive experiments across multiple CL benchmarks and foundation models consistently validate the effectiveness of SD-LoRA.

---

### Paper 52
**Title:** Spurious Forgetting in Continual Learning of Language Models
**Venue:** ICLR 2025 | **Category:** [Analysis]
**Source:** arxiv:2501.13453

**Abstract:**
Recent advancements in large language models (LLMs) reveal a perplexing phenomenon in continual learning: despite extensive training, models experience significant performance declines, raising questions about task alignment and underlying knowledge retention. This study first explores the concept of 'spurious forgetting', proposing that such performance drops often reflect a decline in task alignment rather than true knowledge loss. Through controlled experiments with a synthesized dataset, we investigate the dynamics of model performance during the initial training phases of new tasks, discovering that early optimization steps can disrupt previously established task alignments. Our theoretical analysis connects these shifts to orthogonal updates in model weights, providing a robust framework for understanding this behavior. Ultimately, we introduce a Freezing strategy that fixes the bottom layers of the model, leading to substantial improvements in four continual learning scenarios. Our findings underscore the critical distinction between task alignment and knowledge retention, paving the way for more effective strategies in continual learning.

---

### Paper 53
**Title:** Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning
**Venue:** ICLR 2025 (Oral) | **Category:** [IT]
**Source:** arxiv:2502.11019

**Abstract:**
Catastrophic forgetting (CF) poses a significant challenge in machine learning, where a model forgets previously learned information upon learning new tasks. Despite the advanced capabilities of Large Language Models (LLMs), they continue to face challenges with CF during continual learning. The majority of existing research focuses on analyzing forgetting patterns through a singular training sequence, thereby overlooking the intricate effects that diverse tasks have on model behavior. Our study explores CF across various settings, discovering that model forgetting is influenced by both the specific training tasks and the models themselves. To this end, we interpret forgetting by examining the function vector (FV), a compact representation of functions in LLMs, offering a model-dependent indicator for the occurrence of CF. Through theoretical and empirical analyses, we demonstrated that CF in LLMs primarily stems from biases in function activation rather than the overwriting of task processing functions. Leveraging these insights, we propose a novel function vector guided training methodology, incorporating a regularization technique to stabilize the FV and mitigate forgetting. Empirical tests on four benchmarks confirm the effectiveness of our proposed training method, substantiating our theoretical framework concerning CF and model function dynamics.

---

### Paper 54
**Title:** C-CLIP: Multimodal Continual Learning for Vision-Language Model
**Venue:** ICLR 2025 | **Category:** [VL]
**Source:** OpenReview sb7qHFYwBc

**Abstract:**
Multimodal pre-trained models like CLIP need large image-text pairs for training but often struggle with domain-specific tasks. Since retraining with specialized and historical data incurs significant memory and time costs, it is important to continually learn new domains in the open world while preserving original performance. However, current continual learning research mainly focuses on single-modal scenarios, and the evaluation criteria are insufficient without considering image-text matching performance and the forgetting of zero-shot performance. This work introduces image-caption datasets from various domains and establishes a multimodal vision-language continual learning benchmark. Then, a novel framework named C-CLIP is proposed, which not only prevents forgetting but also enhances new task learning impressively. Comprehensive experiments demonstrate that our method has strong continual learning ability across different domain image-text datasets, and has little forgetting of the original capabilities of zero-shot prediction, significantly outperforming existing methods.

---

### Paper 55
**Title:** Adapt-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection
**Venue:** ICLR 2025 | **Category:** [MM]
**Source:** OpenReview EwFJaXVePU

**Abstract:**
Visual instruction datasets from various distributors are released at different times and often contain a significant number of semantically redundant text-image pairs, depending on their task compositions (i.e., skills) or reference sources. This redundancy greatly limits the efficient deployment of continually adaptable multimodal large language models, hindering their ability to refine existing skills and acquire new competencies over time. To address this, we reframe the problem of lifelong Instruction Tuning (LiIT) via data selection, where the model automatically selects beneficial samples to learn from earlier and new datasets based on the current state of acquired knowledge in the model. Based on empirical analyses that show that selecting the best data subset using a static importance measure is often ineffective for multi-task datasets with evolving distributions, we propose a dynamic data selection framework that adapts its selection criteria as the model learns. Our approach achieves strong forward transfer across the continuum using only a fraction of the original datasets.

---

### Paper 56
**Title:** Vision and Language Synergy for Rehearsal Free Continual Learning
**Venue:** ICLR 2025 | **Category:** [VL]
**Source:** OpenReview 9aZ2ixiYGd

**Abstract:**
The prompt-based approach has demonstrated its success for continual learning problems. However, it still suffers from catastrophic forgetting due to inter-task vector similarity and unfitted new components of previously learned tasks. On the other hand, the language-guided approach falls short of its full potential due to minimum utilized knowledge and participation in the prompt tuning process. To correct this problem, we propose a novel prompt-based structure and algorithm that incorporate 4 key concepts (1) language as input for prompt generation (2) task-wise generators (3) limiting matching descriptors search space via soft task-id prediction (4) generated prompt as auxiliary data. Our experimental analysis shows the superiority of our method to existing SOTAs in CIFAR100, ImageNet-R, and CUB datasets with significant margins i.e. up to 30% final average accuracy, 24% cumulative average accuracy, 8% final forgetting measure, and 7% cumulative forgetting measure.

---

## CVPR 2025 (Papers 57–61)

---

### Paper 57
**Title:** Language Guided Concept Bottleneck Models for Interpretable Continual Learning
**Venue:** CVPR 2025 | **Category:** [VL]
**Source:** arxiv:2503.23283

**Abstract:**
Continual learning (CL) aims to enable learning systems to acquire new knowledge constantly without forgetting previously learned information. CL faces the challenge of mitigating catastrophic forgetting while maintaining interpretability across tasks. Most existing CL methods focus primarily on preserving learned knowledge to improve model performance. However, as new information is introduced, the interpretability of the learning process becomes crucial for understanding the evolving decision-making process, yet it is rarely explored. In this paper, we introduce a novel framework that integrates language-guided Concept Bottleneck Models (CBMs) to address both challenges. Our approach leverages the Concept Bottleneck Layer, aligning semantic consistency with CLIP models to learn human-understandable concepts that can generalize across tasks. By focusing on interpretable concepts, our method not only enhances the model's ability to retain knowledge over time but also provides transparent decision-making insights. We demonstrate the effectiveness of our approach by achieving superior performance on several datasets, outperforming state-of-the-art methods with an improvement of up to 3.06% in final average accuracy on ImageNet-subset. Additionally, we offer concept visualizations for model predictions, further advancing the understanding of interpretable continual learning.

---

### Paper 58
**Title:** AdaDARE-γ: Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adaptation
**Venue:** CVPR 2025 | **Category:** [MM]
**Source:** DOI:10.1109/cvpr52734.2025.01840

**Abstract:**
Adapting Multi-modal Large Language Models (MLLMs) to target tasks often suffers from catastrophic forgetting, where acquiring new task-specific knowledge compromises performance on pre-trained tasks. In this paper, we introduce AdaDARE-γ, an efficient approach that alleviates catastrophic forgetting by controllably injecting new task-specific knowledge through adaptive parameter selection from fine-tuned models without requiring retraining procedures. This approach consists of two key innovations: (1) an adaptive parameter selection mechanism that identifies and retains the most task-relevant parameters from fine-tuned models, and (2) a controlled task-specific information injection strategy that precisely balances the preservation of pre-trained knowledge with the acquisition of new capabilities. Theoretical analysis proves the optimality of our parameter selection strategy and establishes bounds for the task-specific information factor. Extensive experiments on InstructBLIP and LLaVA-1.5 across image captioning and visual question answering tasks demonstrate that AdaDARE-γ establishes state-of-the-art results in balancing model performance. Specifically, it maintains 98.2% of pre-training effectiveness on original tasks while achieving 98.7% of standard fine-tuning performance on target tasks.

---

### Paper 59
**Title:** Synthetic Data is an Elegant GIFT for Continual Vision-Language Models
**Venue:** CVPR 2025 | **Category:** [VL]
**Source:** arxiv:2503.04229

**Abstract:**
Pre-trained Vision-Language Models (VLMs) require Continual Learning (CL) to efficiently update their knowledge and adapt to various downstream tasks without retraining from scratch. However, for VLMs, in addition to the loss of knowledge previously learned from downstream tasks, pre-training knowledge is also corrupted during continual fine-tuning. This issue is exacerbated by the unavailability of original pre-training data, leaving VLM's generalization ability degrading. In this paper, we propose GIFT, a novel continual fine-tuning approach that utilizes synthetic data to overcome catastrophic forgetting in VLMs. Taking advantage of recent advances in text-to-image synthesis, we employ a pre-trained diffusion model to recreate both pre-training and learned downstream task data. In this way, the VLM can revisit previous knowledge through distillation on matching diffusion-generated images and corresponding text prompts. Leveraging the broad distribution and high alignment between synthetic image-text pairs in VLM's feature space, we propose a contrastive distillation loss along with an image-text alignment constraint. To further combat in-distribution overfitting and enhance distillation performance with limited amount of generated data, we incorporate adaptive weight consolidation, utilizing Fisher information from these synthetic image-text pairs and achieving a better stability-plasticity balance. Extensive experiments demonstrate that our method consistently outperforms previous state-of-the-art approaches across various settings.

---

### Paper 60
**Title:** CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning
**Venue:** CVPR 2025 | **Category:** [LoRA]
**Source:** arxiv:2505.24816

**Abstract:**
Class-Incremental Learning (CIL) aims to learn new classes sequentially while retaining the knowledge of previously learned classes. Recently, pre-trained models (PTMs) combined with parameter-efficient fine-tuning (PEFT) have shown remarkable performance in rehearsal-free CIL without requiring exemplars from previous tasks. However, existing adapter-based methods, which incorporate lightweight learnable modules into PTMs for CIL, create new adapters for each new task, leading to both parameter redundancy and failure to leverage shared knowledge across tasks. In this work, we propose ContinuaL Low-Rank Adaptation (CL-LoRA), which introduces a novel dual-adapter architecture combining task-shared adapters to learn cross-task knowledge and task-specific adapters to capture unique features of each new task. Specifically, the shared adapters utilize random orthogonal matrices and leverage knowledge distillation with gradient reassignment to preserve essential shared knowledge. In addition, we introduce learnable block-wise weights for task-specific adapters, which mitigate inter-task interference while maintaining the model's plasticity. We demonstrate CL-LoRA consistently achieves promising performance under multiple benchmarks with reduced training and inference computation, establishing a more efficient and scalable paradigm for continual learning with pre-trained models.

---

### Paper 61
**Title:** LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning
**Venue:** CVPR 2025 | **Category:** [LoRA]
**Source:** arxiv:2503.18985

**Abstract:**
In continual learning (CL), catastrophic forgetting often arises due to feature drift. This challenge is particularly prominent in the exemplar-free continual learning (EFCL) setting, where samples from previous tasks cannot be retained, making it difficult to preserve prior knowledge. To address this issue, some EFCL methods aim to identify feature spaces that minimize the impact on previous tasks while accommodating new ones. However, they rely on static features or outdated statistics stored from old tasks, which prevents them from capturing the dynamic evolution of the feature space in CL, leading to performance degradation over time. In this paper, we introduce the Drift-Resistant Space (DRS), which effectively handles feature drifts without requiring explicit feature modeling or the storage of previous tasks. A novel parameter-efficient fine-tuning approach called Low-Rank Adaptation Subtraction (LoRA-) is proposed to develop the DRS. This method subtracts the LoRA weights of old tasks from the initial pre-trained weight before processing new task data to establish the DRS for model training. Therefore, LoRA- enhances stability, improves efficiency, and simplifies implementation. Furthermore, stabilizing feature drifts allows for better plasticity by learning with a triplet loss. Our method consistently achieves state-of-the-art results, especially for long task sequences, across multiple datasets.
