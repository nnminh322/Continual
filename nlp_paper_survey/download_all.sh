#!/bin/bash
# Download all 109 NLP/LLM CL papers
cd /Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/papers

download() {
  local name="$1"
  local url="$2"
  if [ ! -f "$name" ] || [ $(stat -f%z "$name" 2>/dev/null || echo 0) -lt 50000 ]; then
    curl -sL --max-time 30 -o "$name" "$url"
    local size=$(stat -f%z "$name" 2>/dev/null || echo 0)
    if [ "$size" -gt 50000 ]; then
      echo "OK: $name ($size bytes)"
    else
      echo "FAIL: $name ($size bytes)"
    fi
  else
    echo "SKIP: $name (already exists)"
  fi
}

# NeurIPS 2025
download "02_MINGLE_NeurIPS2025.pdf" "https://openreview.net/pdf/2319ec18a77bdfff982549ce8b4354498ed4e21f.pdf"
download "03_MemEIC_NeurIPS2025.pdf" "https://openreview.net/pdf/e5209240370185240edeb5bc9a5296481fd5702b.pdf"
download "04_MedKnowledgeInjection_NeurIPS2025.pdf" "https://openreview.net/pdf/f58613387e653de73eedc61bbe68e42c49e50e9b.pdf"
download "05_Bisecle_NeurIPS2025.pdf" "https://openreview.net/pdf/801fcac64a080b370564e842cd4b56a2e8974a22.pdf"
download "06_ContinualMultimodalCL_NeurIPS2025.pdf" "https://openreview.net/pdf/69cbc69bcea23845cc672a33eaf78a0dcec49445.pdf"
download "07_DemystifyingLMForgetting_NeurIPS2025.pdf" "https://openreview.net/pdf/067610017dd177dfa6a1029074e8116d3570ba5f.pdf"
download "08_CausalLoRA_NeurIPS2025.pdf" "https://openreview.net/pdf/e9a25c29e826baac254646ad60582c54bfae9652.pdf"
download "09_IntraInterModalForgetting_NeurIPS2025.pdf" "https://openreview.net/pdf/5ad880771d74330c927c6033c68c49e31e0d5ae9.pdf"
download "10_MEMOIR_NeurIPS2025.pdf" "https://openreview.net/pdf/5d7652bd81dac5caf264059bf95575f2c88415e9.pdf"
download "11_SelfEvolvingPseudoRehearsal_NeurIPS2025.pdf" "https://openreview.net/pdf/02bb9435749398313a19d1e2d4389699ebe7d3bf.pdf"
download "12_ReliableLifelongMultimodalEditing_NeurIPS2025.pdf" "https://openreview.net/pdf/e9c3d5a5e890cfc52d58bb0f8c853abe590119fe.pdf"

# ICCV 2025
download "13_MindTheGap_ICCV2025.pdf" "https://arxiv.org/pdf/2507.09118"
download "14_SMoLoRA_ICCV2025.pdf" "https://arxiv.org/pdf/2411.13949"
download "15_DMNSP_ICCV2025.pdf" "https://openaccess.thecvf.com/content/ICCV2025/papers/Kang_Dynamic_Multi-Layer_Null_Space_Projection_for_Vision-Language_Continual_Learning_ICCV_2025_paper.pdf"
download "16_AskAndRemember_ICCV2025.pdf" "https://arxiv.org/pdf/2502.04469"
download "17_TWIST_SCOUT_ICCV2025.pdf" "https://arxiv.org/pdf/2410.10491"
download "18_InstructionGroundedVP_ICCV2025.pdf" "https://arxiv.org/pdf/2508.00260"
download "19_ExternalKnowledgeCLIP_ICCV2025.pdf" "https://arxiv.org/pdf/2503.08510"
download "20_DualDriftVQA_ICCV2025.pdf" "https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Overcoming_Dual_Drift_for_Continual_Long-Tailed_Visual_Question_Answering_ICCV_2025_paper.pdf"
download "21_PLAN_ICCV2025.pdf" "https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_PLAN_Proactive_Low-Rank_Allocation_for_Continual_Learning_ICCV_2025_paper.pdf"

# ACL 2025
download "22_KnowledgeDecoupling_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.646.pdf"
download "23_SerialLifelongEditing_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1492.pdf"
download "24_EfficientDomainCPT_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1578.pdf"
download "25_NSE_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.815.pdf"
download "26_CLoRA_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.940.pdf"
download "27_HiDeLLaVA_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.666.pdf"
download "28_MultiModalityExpansion_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1491.pdf"
download "29_GORP_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.721.pdf"
download "30_DGAR_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.537.pdf"
download "31_LearnToMemorize_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1385.pdf"
download "32_DontHalfListen_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1153.pdf"
download "33_RecurrentKIF_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1328.pdf"
download "34_TiCLM_ACL2025.pdf" "https://aclanthology.org/2025.acl-long.1551.pdf"

# ICML 2025
download "35_FeatureDistributions_ICML2025.pdf" "https://openreview.net/pdf?id=6udKBHc0Mr"
download "36_ReinforcedLifelongEditing_ICML2025.pdf" "https://arxiv.org/pdf/2502.05759"
download "37_LimitsLifelongKE_ICML2025.pdf" "https://arxiv.org/pdf/2503.05683"
download "38_KnowledgeSwapping_ICML2025.pdf" "https://arxiv.org/pdf/2503.05683"
download "39_LearningDynamicsCPT_ICML2025.pdf" "https://arxiv.org/pdf/2505.07796"
download "40_LargeContInstructAssistant_ICML2025.pdf" "https://arxiv.org/pdf/2410.10868"
download "41_TreeLoRA_ICML2025.pdf" "https://arxiv.org/pdf/2506.10355"
download "42_ALKN_ICML2025.pdf" "https://openreview.net/pdf?id=tcK4PV3VN4"
download "43_RAGtoMemory_ICML2025.pdf" "https://arxiv.org/pdf/2502.14802"
download "44_SEFE_ICML2025.pdf" "https://arxiv.org/pdf/2505.02486"
download "45_LADA_ICML2025.pdf" "https://arxiv.org/pdf/2505.23271"
download "46_ComponentialPromptKA_ICML2025.pdf" "https://arxiv.org/pdf/2505.04575"
download "47_ProxyFDA_ICML2025.pdf" "https://arxiv.org/pdf/2505.24088"
download "48_AngleMatters_ICML2025.pdf" "https://openreview.net/pdf?id=6UIer20oYA"

# ICLR 2025
download "49_LOIRE_ICLR2025.pdf" "https://openreview.net/pdf?id=F5PlYMC5ik"
download "50_LLMUnlearning_ICLR2025.pdf" "https://openreview.net/pdf?id=Essg9kb4yx"
download "51_SDLoRA_ICLR2025.pdf" "https://openreview.net/pdf?id=5U1rlpX68A"
download "52_SpuriousForgetting_ICLR2025.pdf" "https://openreview.net/pdf?id=ScI7IlKGdI"
download "53_FunctionVectors_ICLR2025.pdf" "https://openreview.net/pdf?id=gc8QAQfXv6"
download "54_CCLIP_ICLR2025.pdf" "https://openreview.net/pdf?id=sb7qHFYwBc"
download "55_AdaptInf_ICLR2025.pdf" "https://openreview.net/pdf?id=EwFJaXVePU"
download "56_VisionLanguageSynergy_ICLR2025.pdf" "https://openreview.net/pdf?id=9aZ2ixiYGd"

# CVPR 2025
download "57_LanguageGuidedCBM_CVPR2025.pdf" "https://openaccess.thecvf.com//content/CVPR2025/papers/Yu_Language_Guided_Concept_Bottleneck_Models_for_Interpretable_Continual_Learning_CVPR_2025_paper.pdf"
download "58_AdaDARE_CVPR2025.pdf" "https://openaccess.thecvf.com//content/CVPR2025/papers/Xie_AdaDARE-gamma_Balancing_Stability_and_Plasticity_in_Multi-modal_LLMs_through_Efficient_CVPR_2025_paper.pdf"
download "59_SyntheticGIFT_CVPR2025.pdf" "https://arxiv.org/pdf/2503.04229"
download "60_CLLoRA_CVPR2025.pdf" "https://arxiv.org/pdf/2505.24816"
download "61_LoRASubtraction_CVPR2025.pdf" "https://openaccess.thecvf.com//content/CVPR2025/papers/Liu_LoRA_Subtraction_for_Drift_Resistant_Space_in_Exemplar_Free_Continual_Learning_CVPR_2025_paper.pdf"

# NeurIPS 2024
download "62_StabilizingZeroShot_NeurIPS2024.pdf" "https://papers.neurips.cc/paper_files/paper/2024/file/e7feb9dbd9a94b6c552fc403fcebf2ef-Paper-Conference.pdf"
download "63_AdvancingCrossDomain_NeurIPS2024.pdf" "https://openreview.net/pdf/f13992ea7e554b8fcfa2b120be55eeb89c25643f.pdf"
download "64_GlobalAlignment_NeurIPS2024.pdf" "https://openreview.net/pdf/0b2a82c75f549856c3b133f08c9abe7349c018d7.pdf"
download "65_CLAP4CLIP_NeurIPS2024.pdf" "https://openreview.net/pdf/649fc2bc1d6ab7ff1bb07d921e2180c36c2ccf3b.pdf"
download "66_TrainAttention_NeurIPS2024.pdf" "https://openreview.net/pdf/2d2fc4beb4ba2418dd2a4c680959b5708e85b13e.pdf"
download "67_ViLCoBench_NeurIPS2024.pdf" "https://arxiv.org/pdf/2406.13123"
download "68_VPTNullSpace_NeurIPS2024.pdf" "https://arxiv.org/pdf/2406.05658"

# ECCV 2024
download "69_CLIPAdaptiveRepr_ECCV2024.pdf" "https://arxiv.org/pdf/2407.14143"
download "70_MindInterference_ECCV2024.pdf" "https://arxiv.org/pdf/2407.05342"
download "71_SelectDistill_ECCV2024.pdf" "https://arxiv.org/pdf/2403.09296"
download "72_PILoRA_ECCV2024.pdf" "https://arxiv.org/pdf/2401.02094"
download "73_PromptCCD_ECCV2024.pdf" "https://arxiv.org/pdf/2407.19001"
download "74_AnytimeCL_ECCV2024.pdf" "https://arxiv.org/pdf/2409.08518"
download "75_CLIFF_ECCV2024.pdf" "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07221.pdf"
download "76_CLEO_ECCV2024.pdf" "https://arxiv.org/pdf/2407.08411"
download "77_LearnableDriftComp_ECCV2024.pdf" "https://arxiv.org/pdf/2407.08536"
download "78_AdaptWithoutForgetting_ECCV2024.pdf" "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07052.pdf"

# ICML 2024
download "79_COPAL_ICML2024.pdf" "https://openreview.net/pdf?id=Lt8Lk7IQ5b"
download "80_STELLA_ICML2024.pdf" "https://arxiv.org/pdf/2310.08204"

# CVPR 2024
download "81_InfLoRA_CVPR2024.pdf" "https://arxiv.org/pdf/2404.00228"
download "82_MoEAdapters_CVPR2024.pdf" "https://arxiv.org/pdf/2403.11549"
download "83_VLFewShotIL_CVPR2024.pdf" "https://arxiv.org/pdf/2404.02117"
download "84_LanguageGuidedSupervision_CVPR2024.pdf" "https://arxiv.org/pdf/2403.16124"
download "85_TextEnhancedFedCIL_CVPR2024.pdf" "https://arxiv.org/pdf/2403.14101"
download "86_GenMultiModalCIL_CVPR2024.pdf" "https://arxiv.org/pdf/2403.18383"
download "87_ECLIPSE_CVPR2024.pdf" "https://arxiv.org/pdf/2403.20126"

# ICLR 2024
download "88_ScalableLM_ICLR2024.pdf" "https://openreview.net/pdf?id=mz8owj4DXu"
download "89_AdaptLLMReadComp_ICLR2024.pdf" "https://openreview.net/pdf?id=y886UXPEZ0"
download "90_DissectingForgetting_ICLR2024.pdf" "https://openreview.net/pdf?id=tmsqb6WpLz"
download "91_TiCCLIP_ICLR2024.pdf" "https://openreview.net/pdf?id=TLADT8Wrhn"
download "92_CPPO_ICLR2024.pdf" "https://openreview.net/pdf?id=86zAUE80pP"

# AAAI 2024
download "93_TaskAwareLangImg_AAAI2024.pdf" "https://ojs.aaai.org/index.php/AAAI/article/view/28537/29047"
download "94_MaintainingFairness_AAAI2024.pdf" "https://ojs.aaai.org/index.php/AAAI/article/view/33842/36057"

# 2023 papers
download "95_SoftMaskingMixedTasks_EMNLP2023.pdf" "https://arxiv.org/pdf/2310.09436"
download "96_FeCAM_NeurIPS2023.pdf" "https://arxiv.org/pdf/2309.14062"
download "97_ParameterLevelSoftMasking_ICML2023.pdf" "https://arxiv.org/pdf/2306.14775"
download "98_OffDiagonalVL_ICML2023.pdf" "https://arxiv.org/pdf/2305.07437"
download "99_CTP_ICCV2023.pdf" "https://arxiv.org/pdf/2308.07146"
download "100_LanguageGuidedPrompt_ICCV2023.pdf" "https://arxiv.org/pdf/2308.15827"
download "101_PreventZeroShotDeg_ICCV2023.pdf" "https://arxiv.org/pdf/2303.06628"
download "102_MRN_ICCV2023.pdf" "https://arxiv.org/pdf/2305.14758"
download "103_CIGN_ICCV2023.pdf" "https://arxiv.org/pdf/2309.05281"
download "104_ContinualLearningLM_ICLR2023.pdf" "https://openreview.net/pdf?id=m_GDIItaI3o"
download "105_ProgressivePrompts_ICLR2023.pdf" "https://openreview.net/pdf?id=UJTgQBc91_"
download "106_LabelGeneration_ACL2023.pdf" "https://arxiv.org/pdf/2306.12619"
download "107_CrossLingualTransfer_ACL2023.pdf" "https://arxiv.org/pdf/2305.11449"
download "108_ExploringDataGeometry_CVPR2023.pdf" "https://arxiv.org/pdf/2304.03931"
download "109_CODAPrompt_CVPR2023.pdf" "https://arxiv.org/pdf/2211.13218"

echo ""
echo "=== Download Summary ==="
total=$(ls *.pdf 2>/dev/null | wc -l)
ok=$(find . -name "*.pdf" -size +50k | wc -l)
echo "Total files: $total"
echo "Valid PDFs (>50KB): $ok"
echo "Failed/small: $((total - ok))"
