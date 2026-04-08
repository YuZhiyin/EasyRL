from grader import *
save_data = [
    {'PseudoLabel': '42', 'answer': '42'},
    {'PseudoLabel': '1/2', 'answer': '0.5'},
    {'PseudoLabel': '2*x+3', 'answer': '3+2*x'},
]
def calculate_accuracy(save_data):
## 用于带真实标签数据的评估（主要用于验证伪标签质量）
    """Calculate accuracy of pseudo-labels"""
    for i, s in enumerate(save_data):
        if "answer" in s and "PseudoLabel" in s:
            score = 1.0 if s["PseudoLabel"] == s["answer"] else 0.0
            save_data[i]["Accuracy"] = score
    
    print(save_data)
    
def calculate_accuracy(save_data):
    """Calculate accuracy of pseudo-labels using math_equal"""
    for i, s in enumerate(save_data):
        if "answer" in s and "PseudoLabel" in s:
            is_correct = math_equal(s["PseudoLabel"], s["answer"])
            save_data[i]["Accuracy"] = float(is_correct)
    print(save_data)

    
calculate_accuracy(save_data)