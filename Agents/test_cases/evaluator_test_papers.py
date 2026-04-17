# Real paper abstracts for testing the EvaluatorAgent.
# One paper per scoring tier, ordered from NOISE to TRANSFORMATIONAL.
# Expected score ranges are approximate targets for the rubric.

test_papers = [
    # TIER 1 — NOISE (expected: ~1.5-2.5)
    # Contribution: linear LR scaling rule + warmup scheme, no architecture changes
    {
        "title": "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
        "abs": (
            "Deep learning thrives with large neural networks and large datasets. However, larger networks "
            "and larger datasets result in longer training times that impede research and development progress. "
            "Distributed synchronous SGD offers a potential solution to this problem by dividing SGD minibatches "
            "over a pool of parallel workers. Yet to make this scheme efficient, the per-worker workload must be "
            "large, which implies nontrivial growth in the SGD minibatch size. In this paper, we empirically show "
            "that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are "
            "addressed the trained networks exhibit good generalization. Specifically, we show no loss of accuracy "
            "when training with large minibatch sizes up to 8192 images. To achieve this result, we adopt a "
            "hyper-parameter-free linear scaling rule for adjusting learning rates as a function of minibatch size "
            "and develop a new warmup scheme that overcomes optimization challenges early in training."
        ),
        "social": 15
    },

    # TIER 2 — INCREMENTAL (expected: ~3.0-4.5)
    # Contribution: systematic study of ResNet width vs depth, no new architectural idea
    {
        "title": "Wide Residual Networks",
        "abs": (
            "Deep residual networks were shown to be able to scale up to thousands of layers and still have "
            "improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling "
            "the number of layers, and so training very deep residual networks has a problem of diminishing feature "
            "reuse, which makes these networks very slow to train. To tackle these problems, in this paper we "
            "conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose "
            "a novel architecture where we decrease depth and increase width of residual networks. We call the "
            "resulting network structures wide residual networks (WRNs) and show that these are far superior over "
            "their commonly used thin and very deep counterparts. We demonstrate that even a simple 16-layer-deep "
            "wide residual network outperforms in accuracy and efficiency all previous deep residual networks, "
            "including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, "
            "and significant improvements on ImageNet."
        ),
        "social": 60
    },

    # TIER 3 — EVOLUTIONARY (expected: ~5.5-6.5)
    # Contribution: FPN — a new functional module enabling multi-scale feature extraction
    {
        "title": "Feature Pyramid Networks for Object Detection",
        "abs": (
            "Feature pyramids are a basic component in recognition systems for detecting objects at different "
            "scales. But recent deep learning object detectors have avoided pyramid representations, in part "
            "because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, "
            "pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra "
            "cost. A top-down architecture with lateral connections is developed for building high-level semantic "
            "feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows "
            "significant improvement as a generic feature extractor in several applications. Using FPN in a basic "
            "Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection "
            "benchmark without bells and whistles, surpassing all existing single-model entries including those "
            "from the COCO 2016 challenge winners."
        ),
        "social": 350
    },

    # TIER 4 — HIGH IMPACT lower (expected: ~7.1-7.4)
    # Contribution: new transformer encoder-decoder + bipartite matching loss replacing NMS/anchors
    {
        "title": "End-to-End Object Detection with Transformers",
        "abs": (
            "We present a new method that views object detection as a direct set prediction problem. Our approach "
            "streamlines the detection pipeline, effectively removing the need for many hand-designed components "
            "like a non-maximum suppression procedure or anchor generation that explicitly encode our prior "
            "knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or "
            "DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a "
            "transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR "
            "reasons about the relations of the objects and the global image context to directly output the final "
            "set of predictions in parallel. The new model is conceptually simple and does not require a "
            "specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time "
            "performance on par with the well-established and highly-optimized Faster RCNN baseline on the COCO "
            "object detection dataset."
        ),
        "social": 420
    },

    # TIER 4 — HIGH IMPACT upper (expected: ~8.3-8.9)
    # Contribution: entirely new general-purpose backbone replacing ViT/CNN with hierarchical shifted-window attention
    {
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "abs": (
            "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a "
            "general-purpose backbone for computer vision. Challenges in adapting Transformer from language to "
            "vision arise from differences between the two domains, such as large variations in the scale of "
            "visual entities and the high resolution of pixels in images compared to words in text. To address "
            "these differences, we propose a hierarchical Transformer whose representation is computed with "
            "Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention "
            "computation to non-overlapping local windows while also allowing for cross-window connection. This "
            "hierarchical architecture has the flexibility to model at various scales and has linear computational "
            "complexity with respect to image size. These qualities of Swin Transformer make it compatible with a "
            "broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and "
            "dense prediction tasks such as object detection (58.7 box AP) and semantic segmentation "
            "(53.5 mIoU on ADE20K val)."
        ),
        "social": 680
    },

    # TIER 5 — TRANSFORMATIONAL (expected: ~9.0+)
    # Contribution: CLIP — vision-language pretraining enabling zero-shot transfer, paradigm shift away from fixed categories
    {
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "abs": (
            "State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object "
            "categories. This restricted form of supervision limits their generality and usability since additional "
            "labeled data is needed to specify any other visual concept. Learning directly from raw text about "
            "images is a promising alternative which leverages a much broader source of supervision. We demonstrate "
            "that the simple pre-training task of predicting which caption goes with which image is an efficient "
            "and scalable way to learn SOTA image representations from scratch on a dataset of 400 million "
            "image-text pairs collected from the internet. After pre-training, natural language is used to "
            "reference learned visual concepts enabling zero-shot transfer of the model to downstream tasks. The "
            "model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline "
            "without the need for any dataset-specific training. For instance, we match the accuracy of the "
            "original ResNet-50 on ImageNet zero-shot without using any of the 1.28 million training examples it "
            "was trained on."
        ),
        "social": 1200
    },
]
