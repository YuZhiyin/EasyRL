import re
import json
import ast
import gc
import torch
import pandas as pd

"""
Vanilla Inference
"""

QUERY_TEMPLATE_MULTICHOICE = """
Answer the {question_type} question.
{additional_prompt}

Question: 
{question}

{options_str}
""".strip()

# QUERY_TEMPLATE_MATH = """
# <|im_start|>system\nPlease reason step by step, and output your final answer within \\boxed{{}}.<|im_end|>\n
# <|im_start|>user\n{question} Let's think step by step and output the final answer within \\boxed{{}}.<|im_end|>\n
# <|im_start|>assistant\n
# """.strip(" ")

QUERY_TEMPLATE_MATH = """
{question} Let's think step by step and output the final answer within \\boxed{{}}.
""".strip(" ")

# ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\s\n]+)"

def format_option_str(row):
    if not 'options' in row:
        return ''
    options = row['options']
    if type(options) == str:
        options = json.loads(options)
    options = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    options_str = "\n".join(options)
    return f'Options:\n{options_str}\n'

def format_question_vanilla(row):
    question = row['question']
    options_str = format_option_str(row)
    question_type = row['question_type']
    additional_prompt = row['additional_prompt']
    return QUERY_TEMPLATE_MULTICHOICE.format(question=question, options_str=options_str, question_type=question_type, additional_prompt=additional_prompt)

def format_question_math(row):
    question = row['question']
    return QUERY_TEMPLATE_MATH.format(question=question)

def get_the_shortest_str_inlist(str_list):
    return min(str_list, key=len)

def extract_result(res):
    # 如果输入为空或None，返回空字符串
    if not res:
        return ''
    
    match = re.search(ANSWER_PATTERN, res)
    extracted_answer = match.group(1) if match else ''
    # ' res[0].upper()
    # if len(extracted_answer) > 1:
    #     extracted_answer = extracted_answer[0].upper()
    # if not extracted_answer in ['A', 'B', 'C', 'D', 'E']:
    #     return ''
    return extracted_answer

def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def parse_string(s):
    s = s.replace("array(", "").replace(", dtype=object)", "")
    return ast.literal_eval(s)

def clean_string(pl):
    if "' '" in pl or '" "' in pl:
        pl = pl.strip('[]').split("' '")
        pl = [item.strip("' ") for item in pl]
    if isinstance(pl, str) and pl.startswith('['):
        pl = parse_string(pl)
    if isinstance(pl, list):
        pl = get_the_shortest_str_inlist(pl)
    pl = pl.replace('"', '').replace("'", '').replace('[]', '')
    pl = pl.replace('.', '').strip()

    return pl

def pack_answer(row):
    if 'PseudoLabel' in row:
        pl = row['PseudoLabel']
    else:
        pl = row['answer']

    if type(pl) != str:
        pl = parse_string(row['answers_spans'])['spans'][0]
    
    pl = clean_string(pl)
    return f'Answer: {pl}'

"""
Check Answer
"""

def check_consistency(res, gt):
    pred = extract_result(res)
    gt_pred = extract_result(gt)
    return pred == gt_pred

def check_answer(res, gt):
    pred = extract_result(res)
    # if length not same, cut to the same length
    if len(pred) < len(gt):
        pred = pred[:len(gt)]
    elif len(pred) > len(gt):
        gt = gt[:len(pred)]
    return pred == gt

# for value inference
def normoalize_num(num):
    def eval_num(num):
        num = num.replace('%','/100').replace(',','')
        try:
            num = eval(num)
        except Exception as e:
            num = float('inf')
            pass
        return num
    VALUE_PATTERRN = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?[%]*"
    val_reg = re.compile(VALUE_PATTERRN)
    return [eval_num(num) for num in val_reg.findall(num)]


def check_value_equal(res_arr, gt_arr):
    import math
    for gt_num in gt_arr:
        for pred_num in res_arr:
            if math.isclose(pred_num, gt_num, rel_tol=1e-2):
                return True
    return False

def check_answer_value(res, gt):
    pred = normoalize_num(extract_result(res))
    gt = normoalize_num(gt)
    return check_value_equal(pred, gt)

def exact_match(pred, answer):
    return pred == answer

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    import string
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1

def check_answer_fuzzy(res: str, gt: list):
    pred = extract_result(res)
    match_list = [fuzzy_match(pred, gt_item) for gt_item in gt]
    return True in match_list

def convert_to_conversation_format(row, task_config):
    # Get question_type and additional_prompt from task config
    row['question_type'] = task_config['question_type']
    row['additional_prompt'] = task_config['additional_prompt']
    
    messages = [
        {"role": "user", "content": format_question_vanilla(row)},
        {"role": "assistant", "content": pack_answer(row)}
    ]

    if "system_prompt" in task_config:
        messages.insert(0, {"role": "system", "content": ""})

    return messages

def format_question_alpaca(row, format_fn=format_question_vanilla, task_config=None):
    row['question_type'] = task_config['question_type']
    row['additional_prompt'] = task_config['additional_prompt']
    input_text = format_fn(row)
    output_test = pack_answer(row)
    return {
        "instruction": input_text,
        "input": '',
        "output": output_test
    }

def clear_mem(verbose: bool = False) -> None:
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()

    def try_attr(x, a):
        try:
            return getattr(x, a)
        except Exception:
            return None

    if verbose:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)

    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

FEW_SHOT_SYSTEM = """
You are an expert in the multiple choice question. Below are some examples of questions and their corresponding answer.

{reference}
""".strip()

REFLECTION = """Here are the multiple answers of the multiple choice question.  Please consider them thoroughly and give me the correct answer. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}

Options:
{options}

Multiple Answers:
{answers}

Now, please directly give me the final correct answer:
"""

# REFLECTION_MATH = """Here are the multiple answers of the math question.  Please consider them thoroughly and give me the correct answer. Output the final answer within \\boxed{{}}.

# Question: 
# {question}

# Multiple Answers:
# {answers}

# Now, please directly give me the final correct answer:
# """

REFLECTION_MATH = """You are given multiple proposed answers to a math problem.  
Your task is to carefully examine these answers and determine whether any of them is correct.

- If one of the proposed answers is correct, return it as the final answer.
- If **none** of the proposed answers is correct, **re-solve the problem step-by-step** and provide the correct answer.
- Always show the **final answer** clearly inside \\boxed{{}}.

Question:
{question}

Proposed Answers:2,2

Now, please reflect on the answers above and give the final correct answer in \\boxed{{}}.
"""

def format_reflection(data):
    question = data['question']
    options = data['options']
    preds = '\n'.join(data['Preds'])
    return REFLECTION.format(question=question, options=options, answers=preds)

# def format_reflection_math(data):
#     question = data['question']
#     preds = '\n'.join(data['PredAnswers'])
#     return REFLECTION_MATH.format(question=question, answers=preds)

def format_reflection_math(data):
    question = data['question']
    # preds = '\n'.join(data['PredAnswers'])
    return REFLECTION_MATH.format(question=question)

