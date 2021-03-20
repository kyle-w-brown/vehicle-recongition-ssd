{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "nvidia_deeplearningexamples_ssd.ipynb",
      "provenance": []
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tJ6Da0BSfzzE"
      },
      "source": [
        "### This notebook requires a GPU runtime to run.\n",
        "### Please select the menu option \"Runtime\" -> \"Change runtime type\", select \"Hardware Accelerator\" -> \"GPU\" and click \"SAVE\"\n",
        "\n",
        "----------------------------------------------------------------------\n",
        "\n",
        "# SSD\n",
        "\n",
        "*Author: NVIDIA*\n",
        "\n",
        "**Single Shot MultiBox Detector model for object detection**\n",
        "\n",
        "_ | _\n",
        "- | -\n",
        "![alt](https://pytorch.org/assets/images/ssd_diagram.png) | ![alt](https://pytorch.org/assets/images/ssd.png)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "znqNSRPlfzzV"
      },
      "source": [
        "import torch\n",
        "precision = 'fp32'\n",
        "ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IofKBW23fzzX"
      },
      "source": [
        "will load an SSD model pretrained on COCO dataset from Torch Hub.\n",
        "\n",
        "Setting precision='fp16' will load a checkpoint trained with [mixed precision](https://arxiv.org/abs/1710.03740) into architecture enabling execution on [Tensor Cores](https://developer.nvidia.com/tensor-cores).\n",
        "Handling mixed precision data requires [Apex](https://github.com/NVIDIA/apex) library.\n",
        "\n",
        "\n",
        "\n",
        "### Model Description\n",
        "\n",
        "This SSD300 model is based on the\n",
        "[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper, which\n",
        "describes SSD as “a method for detecting objects in images using a single deep neural network\".\n",
        "The input size is fixed to 300x300.\n",
        "\n",
        "The main difference between this model and the one described in the paper is in the backbone.\n",
        "Specifically, the VGG model is obsolete and is replaced by the ResNet-50 model.\n",
        "\n",
        "From the\n",
        "[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012)\n",
        "paper, the following enhancements were made to the backbone:\n",
        "*   The conv5_x, avgpool, fc and softmax layers were removed from the original classification model.\n",
        "*   All strides in conv4_x are set to 1x1.\n",
        "\n",
        "The backbone is followed by 5 additional convolutional layers.\n",
        "In addition to the convolutional layers, we attached 6 detection heads:\n",
        "*   The first detection head is attached to the last conv4_x layer.\n",
        "*   The other five detection heads are attached to the corresponding 5 additional layers.\n",
        "\n",
        "Detector heads are similar to the ones referenced in the paper, however,\n",
        "they are enhanced by additional BatchNorm layers after each convolution.\n",
        "\n",
        "### Example\n",
        "\n",
        "In the example below we will use the pretrained SSD model loaded from Torch Hub to detect objects in sample images and visualize the result.\n",
        "\n",
        "To run the example you need some extra python packages installed.\n",
        "These are needed for preprocessing images and visualization."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GDjO9nlZfzzZ"
      },
      "source": [
        "%%bash\n",
        "pip install numpy scipy scikit-image matplotlib"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HVPSdC2jfzza"
      },
      "source": [
        "For convenient and comprehensive formatting of input and output of the model, load a set of utility methods."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zn-DAZmnfzzb"
      },
      "source": [
        "utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2EjjBVXHfzzc"
      },
      "source": [
        "Now, prepare the loaded model for inference"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pIw98GoUfzzd"
      },
      "source": [
        "ssd_model.to('cuda')\n",
        "ssd_model.eval()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q74vUW_Qfzze"
      },
      "source": [
        "Prepare input images for object detection.\n",
        "(Example links below correspond to first few test images from the COCO dataset, but you can also specify paths to your local images here)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-YT1IT1Hfzzf"
      },
      "source": [
        "uris = [\n",
        "    'http://images.cocodataset.org/val2017/000000397133.jpg',\n",
        "    'http://images.cocodataset.org/val2017/000000037777.jpg',\n",
        "    'http://images.cocodataset.org/val2017/000000252219.jpg'\n",
        "]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GkjIQ5QGfzzg"
      },
      "source": [
        "Format the images to comply with the network input and convert them to tensor."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lzf70Nk9fzzg"
      },
      "source": [
        "inputs = [utils.prepare_input(uri) for uri in uris]\n",
        "tensor = utils.prepare_tensor(inputs, precision == 'fp16')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Nxk4emg_fzzh"
      },
      "source": [
        "Run the SSD network to perform object detection."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TZyoyVTnfzzh"
      },
      "source": [
        "with torch.no_grad():\n",
        "    detections_batch = ssd_model(tensor)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KC-EiFCRfzzi"
      },
      "source": [
        "By default, raw output from SSD network per input image contains\n",
        "8732 boxes with localization and class probability distribution.\n",
        "Let's filter this output to only get reasonable detections (confidence>40%) in a more comprehensive format."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "phQsbhFafzzi"
      },
      "source": [
        "results_per_input = utils.decode_results(detections_batch)\n",
        "best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hc-bVOUKfzzj"
      },
      "source": [
        "The model was trained on COCO dataset, which we need to access in order to translate class IDs into object names.\n",
        "For the first time, downloading annotations may take a while."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EEsUnnM6fzzj"
      },
      "source": [
        "classes_to_labels = utils.get_coco_object_dictionary()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-9PSN_sJfzzk"
      },
      "source": [
        "Finally, let's visualize our detections"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vpfm649Jfzzk"
      },
      "source": [
        "from matplotlib import pyplot as plt\n",
        "import matplotlib.patches as patches\n",
        "\n",
        "for image_idx in range(len(best_results_per_input)):\n",
        "    fig, ax = plt.subplots(1)\n",
        "    # Show original, denormalized image...\n",
        "    image = inputs[image_idx] / 2 + 0.5\n",
        "    ax.imshow(image)\n",
        "    # ...with detections\n",
        "    bboxes, classes, confidences = best_results_per_input[image_idx]\n",
        "    for idx in range(len(bboxes)):\n",
        "        left, bot, right, top = bboxes[idx]\n",
        "        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]\n",
        "        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')\n",
        "        ax.add_patch(rect)\n",
        "        ax.text(x, y, \"{} {:.0f}%\".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-MthATmwfzzm"
      },
      "source": [
        "### Details\n",
        "For detailed information on model input and output,\n",
        "training recipies, inference and performance visit:\n",
        "[github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)\n",
        "and/or [NGC](https://ngc.nvidia.com/catalog/model-scripts/nvidia:ssd_for_pytorch)\n",
        "\n",
        "### References\n",
        "\n",
        " - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper\n",
        " - [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012) paper\n",
        " - [SSD on NGC](https://ngc.nvidia.com/catalog/model-scripts/nvidia:ssd_for_pytorch)\n",
        " - [SSD on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)"
      ]
    }
  ]
}