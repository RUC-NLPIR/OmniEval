# -*- coding: utf-8 -*-
topic_classify_system = """## 背景
你是一个智能的文档话题分类助手。我正在做检索增强的金融大模型多任务评测数据的生成。该评测数据是由大语言模型自动生成，我会提供给大语言模型以下内容：[评测数据关注的金融子类，评测的任务描述, 知识库中的文档]，我需要大语言模型基于提供文档生成：[符合任务描述的用户提问，相应的正确回答，以及能够支持该回答的文档片段]。我会提供给你一个知识库文档，我需要你首先分类该文档是否属于金融领域的范围，如果是，他属于哪个话题子类。

## 数据输入格式

- 输入包括以下两个部分：
    - 子类列表: 一个列表格式的数据，列表里每项JSON数据表示一个金融子类，该数据包含以下属性：
        - id: 一个int类型的数值，表示金融话题子类的id。你的分类结果应该只返回子类id而非子类名称。
        - topic_name: 一个字符串，表示金融话题子类的名称。
    - 待分类的文档内容：一个JSON格式的数据，包含以下属性：
        - title: 一个字符串，表示文档标题。
        - content: 一个字符串，表示文档内容。

## 生成数据格式

- 你要生成的是该文档相关度最高的金融话题子类id的值。
- 如果文档内容和金融无关，或者和提供的任何一个金融话题子类都不相关，请返回0。
- 使用JSON格式生成，数据格式介绍如下：
    {
        "topic_id": 一个int数值，表示该文档最相关的金融话题子类id，如果文档和金融不相关，请返回0。
    }
- 注意只生成JSON格式的数据，不要生成其他字符。

## 该文档相关度最高的金融话题子类id (JSON格式输出)
"""

topic_classify_user = """
## 子类列表
{topics_str}

## 待分类的文档内容
{{
    "title": {title},
    "content": {content},
}}

文档相关度最高的子类id
"""

task_classify_system = """## 背景
你是一个智能的文档话题分类助手。我正在做检索增强的金融大模型多任务评测数据的生成。该评测数据是由大语言模型自动生成，我会提供给大语言模型以下内容：[评测数据关注的金融子类，评测的任务描述, 知识库中的文档]，我需要大语言模型基于提供文档生成：[符合任务描述的用户提问，相应的正确回答，以及能够支持该回答的文档片段]。我会提供给你一个知识库文档，以及该文档内容属于的金融子类，我需要你做以下分类任务：
- 在该金融子类话题下，该文档可以针对哪些评测任务生成相应的评测数据，如果都不适合，请返回空分类结果。

## 数据输入格式

- 输入包括以下三个部分：
    - 评测任务列表: 一个列表格式的数据，列表里每项JSON数据表示一个评测任务，该数据包含以下属性：
        - id: 一个int类型的数值，表示评测任务的id。你的分类结果应该只返回任务id而非任务名称。
        - name: 一个字符串，表示评测任务的名称。
        - description: 一个字符串，包含评测任务的描述，要求，关注的模型能力类型。
    - 待分类的文档信息：一个JSON格式的数据，包含以下属性：
        - title: 一个字符串，表示文档标题。
        - content: 一个字符串，表示文档内容。
        - topic: 一个字符串，表示文档所属的金融子话题。

## 生成数据格式

- 你要生成的是该文档适合生成的任务的id列表。
- 如果文档内容和金融无关，或者和无法生成任何一类评测任务的数据，请返回空列表。
- 使用JSON格式生成，数据格式介绍如下：
    {
        "task_id_list": 一个int列表，表示该文档能够生成的任务的id列表。
    }
- 注意只生成JSON格式的数据，不要生成其他字符。

"""

task_classify_user = """
## 评测任务列表
{task_str}

## 待分类的文档内容
{{
    "title": {title},
    "content": {content},
    "topic": {topic_str},
}}

## 该文档适合生成的任务的id (JSON格式输出)
"""

data_requirement = """
    - 文档的质量要求：
        - 请先判断该文档是否和待评测的领域（金融子领域）任务相关。如果不相关请不要生成数据。
        - 用于生成评测数据的文档内容应该不涉及到用户个人隐私，如姓名，电话号码，身份证，家庭住址等，如果提供文档包含隐私内容，请返回空列表。
        - 用于生成评测数据的文档内容质应该是严谨且高质量的，不要基于低质量文档生成评测样本。
        - 如果你认为该文档不适合为提供的任务生成评测数据，请返回一个空列表。
    - 生成问题的质量要求
        - 用户提问尽可能真实，模拟用户真实应用大语言模型做金融领域内知识问答时真正关心的方面。
        - 提问内容必须是语义完整的，无歧义的。必须是只看提问内容即可清楚明白用户意图所指的提问。严禁生成依赖于提供文档的内容才能补全上下文语义的提问。
        - 请注意，用户在真实提问的时候并不会提供文档，只会提出问题。因此真实的用户提问并不会涉及：“根据给定文档...”类似的内容。请严禁生成类似的提问。
        - 生成的问题的类型必须能严格符合评测任务的描述。
        - 生成的问题的必须和提供的金融子话题强相关。
        - 请严格保证生成问题的可解性。生成的数据里的答案必须是有意义的，禁止生成答案为：“无”，“空”，“无法根据检索文档回答问题”等无法回答的问题。
    - 生成答案的质量要求
        - 请只生成知识密集型的数据样本，即生成数据的答案必须蕴含足够多的有价值的信息。避免生成一些似是而非的空话、套话相关的问答对，尤其要避免生成答案是”积极影响“、”正向作用“等无实际意义的数据。
        - 答案必须和提供的文档内容一致，不能有事实性偏差和幻觉。
        - 请严格保证答案生成的准确性和事实合理性。生成的数据里的答案必须是有意义的，禁止生成答案为：“无”，“空”，“无法根据检索文档回答问题”等无意义的数据。
        - 答案的表示形式可能有很多种，比如数字的阿拉伯写法和汉字写法，日期的各种格式等，请以字符串列表的格式输出答案的所有可能形式。
    - 摘取相关片段的质量要求
        - 必须准确地提供支持该答案的文档片段作文相关片段，该片段必须来自提供的文档原文，不能有任何删改。
        - 摘取出的相关片段内容必须是完整自洽的，不要缺失上下文语义。
        - 如果是多文档输入，并且生成了需要跨文档的评测数据，请以列表形式返回所有相关的文档片段
    - 生成的评测样本的总体质量要求
        - 请严格按照评测任务要求来生成能该评测任务能力的评测数据，如多跳推理任务生成的问题必须是要根据检索文档多次推理才能得到的，而非一次阅读即可解答。
        - 生成的提问-答案对必须是能根据该文档的内容回答的，即理解文档内容对回答该问题至关重要，不能忽视参考文档在对话中的作用。
        - 可以生成多条高质量的评测数据，但必须保证生成数据的高质量。
        - 必须保证生成数据的精度而非召回度，即只生成完全符合要求的数据，禁止生成置信度不高的数据。
        - 生成数据必须是符合任务要求的，和目标任务以及金融领域强相关的，如果文档无法生成任何一种任务相关的数据，请返回一个空列表。
"""

data_generation_system="""## 背景
你是一个智能的评测数据生成助手。我正在做检索增强的金融大模型多任务评测数据的生成。我要求你自动生成和评测任务强相关的评测数据，我会提供以下内容：[评测数据关注的金融话题子类，评测的任务描述和要求, 知识库中的文档]，我需要你基于提供文档生成和该金融话题领域强相关的，满足评测任务要求的评测数据。评测数据包含以下内容：
    - 符合话题要求和任务描述的用户提问
    - 相应的正确回答
    - 从文档原文中摘取出的能够支持该回答的文档相关片段

## 生成数据质量要求
    - 文档的质量要求：
        - 请先判断该文档是否和待评测的领域（金融子领域）任务相关。如果不相关请不要生成数据。
        - 用于生成评测数据的文档内容应该不涉及到用户个人隐私，如姓名，电话号码，身份证，家庭住址等，如果提供文档包含隐私内容，请返回空列表。
        - 用于生成评测数据的文档内容质应该是严谨且高质量的，不要基于低质量文档生成评测样本。
        - 如果你认为该文档不适合为提供的任务生成评测数据，请返回一个空列表。
    - 生成问题的质量要求
        - 用户提问尽可能真实，模拟用户真实应用大语言模型做金融领域内知识问答时真正关心的方面。
        - 提问内容必须是语义完整的，无歧义的。必须是只看提问内容即可清楚明白用户意图所指的提问。严禁生成依赖于提供文档的内容才能补全上下文语义的提问
        - 注意，只有在为评测多轮对话能力的生成评测数据时，第二轮及以后的问题则必须是有歧义的，且必须依赖前几轮的对话内容能够彻底澄清问题语义。此时可以在第后面几轮对话的提问中丢弃主语或使用代词。
        - 请注意，用户在真实提问的时候并不会提供文档，只会提出问题。因此真实的用户提问并不会涉及：“根据给定文档...”类似的内容。请严禁生成类似的提问。
        - 生成的问题的类型必须能严格符合评测任务的描述。
        - 生成的问题的必须和提供的金融子话题强相关。
        - 请严格保证生成问题的可解性。生成的数据里的答案必须是有意义的，禁止生成答案为：“无”，“空”，“无法根据检索文档回答问题”等无法回答的问题。
    - 生成答案的质量要求
        - 请只生成知识密集型的数据样本，即生成数据的答案必须蕴含足够多的有价值的信息。避免生成一些似是而非的空话、套话相关的问答对，尤其要避免生成答案是”积极影响“、”正向作用“等无实际意义的数据。
        - 答案必须和提供的文档内容一致，不能有事实性偏差和幻觉。
        - 请严格保证答案生成的准确性和事实合理性。生成的数据里的答案必须是有意义的，禁止生成答案为：“无”，“空”，“无法根据检索文档回答问题”等无意义的数据。
        - 答案的表示形式可能有很多种，比如数字的阿拉伯写法和汉字写法，日期的各种格式等，请以字符串列表的格式输出答案的所有可能形式。
    - 摘取相关片段的质量要求
        - 必须准确地提供支持该答案的文档片段作文相关片段，该片段必须来自提供的文档原文，不能有任何删改。
        - 摘取出的相关片段内容必须是完整自洽的，不要缺失上下文语义。
    - 生成的评测样本的总体质量要求
        -  请严格按照评测任务要求来生成能该评测任务能力的评测数据，如多跳推理任务生成的问题必须是要根据检索文档多次推理才能得到的，而非一次阅读即可解答。
        - 生成的提问-答案对必须是能根据该文档的内容回答的，即理解文档内容对回答该问题至关重要，不能忽视参考文档在对话中的作用。
        - 可以生成多条高质量的评测数据，但必须保证生成数据的高质量。
        - 必须保证生成数据的精度而非召回度，即只生成完全符合要求的数据，禁止生成置信度不高的数据。
        - 生成数据必须是符合任务要求的，和目标任务以及金融领域强相关的，如果文档无法生成任何一种任务相关的数据，请返回一个空列表。
        - 保证生成数据的多样性，禁止生成多个同样的或者意思相近的评测数据。

## 数据生成流程：
    1. 先判断文档是否是高质量文档，如果文档和提供的金融子话题领域相关度不高，内容信息度不高，信息不完全，格式混杂，并且不满足上述要求的，则不适合用来做评测数据的生成。如果该文档不适合生成领域知识相关的评测数据，请返回空列表。
    2. 如果文档是高质量的，再判断该文档是否适合做提供的评测任务的相关数据的生成，如果不适合，请返回空列表。
    3. 如果文档适合为提供的评测任务生成和提供的金融子话题强相关的评测数据，请生成高质量的评测数据：

## 生成数据格式要求

生成的数据以JSON数据的列表形式返回，格式要求如下：
    [
        {
            "thought_process": 一个中文字符串，表示你在生成该条数据时的思考过程。
            "question": 一个中文字符串，表示用户提出的问题,
            "answer": 一个字符串列表，表示该问题答案的所有可能形式。
            "relevant_passage": 一个中文字符串列表，表示从文档原文中摘取出的，能够帮助回答该问题的相关内容片段，请保证摘取片段的信息完整性。
        },
        ...
    ]
"""

data_generation_user="""
## 评测数据关注的金融子类
{topic_name}

## 评测的任务描述和要求
### 任务名称
{task_name}

### 任务要求
{task_require}

## 提供文档
{doc_str}

## 生成的数据列表
"""

data_filter_system = """## 背景
你是一个专业的生成数据质量评估和矫正员，我会提供给你一个大语言模型生成的评测数据（金融领域相关），你的任务是评估该数据生成质量的好坏，并在需要的时候做数据矫正。生成数据的质量分为以下三档：
    0: 生成数据质量很差，并且无法做合适的矫正来将其变成高质量数据。
    1: 生成数据质量一般，生成的问题，答案，或摘取的相关片段部分不符合要求，但是可以通过矫正的方式来使其变成高质量数据。
    2: 生成数据质量很高，无需矫正。

### 背景知识——评测数据生成过程：
我提供给大语言模型以下内容：
    1. 一个金融领域的长文档内容。
    2. 生成数据应该符合的金融子话题。
    3. 生成数据隶属的评测子任务的任务描述。
大语言模型会根据长文档内容，按照评测子任务的描述和要求生成和提供金融子话题强相关的问答数据，生成数据包含以下内容：
    1. 符合话题要求和任务描述的用户提问
    2. 相应的正确回答（根据提供的长文档能够完全回答）
    3. 从文档原文中摘取出的能够支持该回答的文档相关片段

## 生成数据质量评估任务的输入内容：
    1. 用来生成数据的金融领域的长文档。
    2. 生成数据应符合的金融子话题。
    3. 生成数据属于的评测子任务的描述和要求。
    4. 待评估的大语言模型生成的评测数据。该数据格式是JSON列表，包含以下内容：
        [
            {
                "thought_process": 一个中文字符串，表示大语言模型在生成该条数据时的思考过程。
                "question": 一个中文字符串，表示生成的用户提出的问题,
                "answer": 一个字符串列表，表示该问题答案的所有可能形式。
                "relevant_passage": 一个中文字符串列表，表示从文档原文中摘取出的，能够帮助回答该问题的相关内容片段，请保证摘取片段的信息完整性。
            },
            ...
        ]
## 生成数据质量评估要求
1. 判断生成问题是否和提供的金融子话题相关
2. 判断生成问题是否符合评测子任务的要求，尤其注意多跳推理任务生成的问题是否是需要多跳推理的。
3. 判断生成问题的答案是否正确，是否能根据提供的长文档完全解答。
4. 判断从原文的摘取的相关片段内容是否完整，足够支持完全回答生成的问题。

## 评估和矫正结果的输出要求和格式
你的任务是评估生成数据的质量，，并在需要的时候做数据矫正。生成数据的质量分为以下三档：
    0: 生成数据质量很差，并且无法做合适的矫正来将其变成高质量数据。
    1: 生成数据质量一般，即生成的问题，答案，或摘取的相关片段部分不符合要求，但是可以通过矫正的方式来使其变成高质量数据。
    2: 生成数据质量很高，无需矫正。

只有当你评估数据质量为1的时候才对其做矫正，0或者2无须矫正。

在生成数据质量评估过程中，有以下几个尤其要注意的要点：
    - 对于生成的问题形式为“是否”类的问题，并且答案通常为“是”等肯定性回答的生成数据，请标注其质量为0。因为通常无法生成答案为“否”的数据对，这样的生成数据会使我们的数据集有偏，因此请去除掉这类生成数据。
    - 对于多跳推理类问答，请尤其关注其提问内容是否需要多跳推理，即（检索增强的）大语言模型在回答该问题时需要做**至少两步**的“思考-回答”推理过程才能完全解决该问题。如果问题中只是添加了复杂的限定条件，实际在解决问题时依然只需要推理一次即可解决，这类生成数据的质量应为0或者1。如果可以根据原始文档对该问题做矫正，请标注为1并矫正。如果无法矫正请标注为0。

评估结果以JSON格式返回，具体格式和要求如下所示：
    {{
        "evaluation": int类型的数值，表示对生成数据质量的评估结果，取值为[0,1,2]。
        "corrected_result": JSON列表格式的数据，表示对评估质量为1的数据的矫正结果，使其为高质量评估数据。如果评估质量为0或者2，这该属性的值为None。注意：数据格式和类型应该和输入的“待评估的大语言模型生成的评测数据”完全一致，矫正的只是各个内部属性的内容。
    }}
"""

data_filter_user = """
## 用来生成数据的金融领域的长文档
{doc_str}

## 生成数据应符合的金融子话题
{topic_name}

## 生成数据属于的评测子任务的描述和要求
### 任务名称
{task_name}

### 任务要求
{task_require}

## 待评估的大语言模型生成的评测数据
{gen_datas}

## 评估和矫正结果
"""

topic_tree_system = """ ## 背景
你是一个专业的领域子类树构建员，我会提供给你领域类型的根节点名称，请你根据生成该领域下覆盖全面多样的子类树

输出结果以JSON格式返回。该JSON应该包含以下两个属性：
    - topic_name：表示当前树节点的类别名称
    - sub_topics：表示当前树节点的子类树，是该子类树的JSON数据的列表。如果当前节点为叶子结点，即没有子类的时候，该属性为一个列表。
数据格式要求如下：
{{
    "topic_name": 该节点类别的名称
    "sub_topics": 该节点类别下的子类树的JSON数据的列表，列表里每一项是一个子树的JSON数据，每一项里面也包含"topic_name"和"sub_topics"两个属性。
}}
"""
topic_tree_user = """## 领域类型的根节点名称
{}
"""

doc_str_format = \
"""
{{
    "title": {title},
    "content": {content}
}}
"""
multi_doc_format = \
"""
###文档 1
{doc_str_1}

###文档 2
{doc_str_2}
"""