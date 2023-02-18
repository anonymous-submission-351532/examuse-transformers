# EAM: Experimental Analysis of Structural Embeddings for MIDI-based Symbolic Music Generation

## Abstract

Extending widely used large language models, recent work have proposed various adaptations of large-scale Transformer architectures to music generation. However, existing work share two major drawbacks: 1) their tokenization layers require domain-specific structural information (e.g. bars and beats) that are typically missing in in-the-wild data or 2) they lack reproducibility with no publicly available implementation. In light of such limitations, we implement a MIDI-based music generation framework inspired by MuseNet, and empirically study the effect of different structural embedding setups while only using information available in in-the-wild data. Experimental analyses under various metrics reveal interesting insights that provide guidance toward which structural encoding setup to deploy. These insights may shed light on improving music generation using the existing architecture of the common language model. In addition, various embedding setups may allow the users to create music in different aspects. We open-source our implementation via [HuggingFace](https://github.com/anonymous) towards reproducibility for practitioners upon acceptance.

![method_figure.png](https://huggingface.co/acl-submission-anonym/EAM-relative/resolve/main/method_figure.png)

## Install

The code block below shows the actual commands to install the Examuse-Transformers package. When you execute this code, the latest version of the package will be downloaded from GitHub, and a folder named "examuse-transformers" will be created in the current directory. Then, move to this folder and check out the version tag named **`${VERSION}`**. Finally, to install this package locally, run the command **`pip install --editable .`**.

The **`--editable`** option is used to install the package locally so that any changes to the source code can be immediately reflected.

```bash
$git clone https://github.com/anonymous-submission-351532/examuse-transformers.git
$cd examuse-transformers
$git checkout tags/${VERSION}
$pip install --editable .
```

## Usage

```python
>> from transformers import AutoModelForCausalLM, AutoTokenizer
>> model_name = "acl-submission-anonym/EAM-spectral"
>> # if you want relative model, see here: "acl-submission-anonym/EAM-relative"
>> model = AutoModelForCausalLM.from_pretrained(model_name) 
>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Model Cards

|  | GPT2-based model |
| --- | --- |
| Relative Model | https://huggingface.co/acl-submission-anonym/EAM-relative |
| Spectral Model | https://huggingface.co/acl-submission-anonym/EAM-spectral |

## Run on Specific(Updated) HF Versions

Here's a guide on how to rebase a forked library 'EMA' from a repository 'transformers' to a specific version if 'transformers' is updated:

1. Add the original repository 'transformers' as a remote:
    
    ```
    $ git remote add upstream https://github.com/huggingface/transformers.git
    ```
    
2. Fetch the changes from 'transformers':
    
    ```
    $ git fetch upstream
    ```
    
3. Create a new branch to work on:
    
    ```
    $ git checkout examuse
    $ git checkout -b ${new-feature-branch}
    ```
    
4. Rebase the new branch on top of the latest changes from 'transformers':
    
    ```
    $ git rebase upstream/tags/<version_name>
    ```
    
    This will replay your changes on top of the latest changes from 'transformers'.
    
5. Resolve any conflicts that arise during the rebase process.

Now you have successfully rebased your forked repository 'EMA' to the latest changes from 'transformers'. Repeat this process whenever you want to update your forked repository to a new version of 'transformers'.

## Reference

TBA