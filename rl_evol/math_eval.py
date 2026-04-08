from math_verify import parse, verify

# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use
# parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

gold = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")
print(gold)
print(answer)
# Order here is important!
verify(gold, answer)
result = verify(gold, answer)
print(result)
# >>> True