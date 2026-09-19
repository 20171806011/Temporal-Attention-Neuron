"""Independent analysis of Sprint 4.3-A composition probe computation carrier and data flow.

Answers the 5 core questions:
1. Does composition output depend on step-by-step membrane potential updates?
2. Does it pass through spike/reset?
3. Is it a static retriever plus explicit arithmetic node?
4. Which checks are actual numerical verifications vs hardcoded True flags?
5. Does the result prove operator sufficiency, functional correctness, online neural dynamics, or learned generalization?
"""
import ast
import os
import json

def verify_carrier_assertions(results):
    assert not results['run_history_analysis']['has_membrane_potential_h'], "Expected no membrane potential h in run_history"
    assert not results['run_history_analysis']['has_spike_or_reset'], "Expected no spike or reset in run_history"
    assert results['run_history_analysis']['direct_arithmetic_found'], "Expected direct explicit arithmetic in run_history"
    assert results['answers']['hardcoded_true_count_shortcut'] == 5, f"Expected 5 True in shortcut dict, got {results['answers']['hardcoded_true_count_shortcut']}"
    assert results['answers']['direct_check_true_count'] == 1, f"Expected 1 direct check(..., True), got {results['answers']['direct_check_true_count']}"
    assert results['answers']['q4_hardcoded_true_checks_count'] == 6, f"Expected 6 total structural True checks, got {results['answers']['q4_hardcoded_true_checks_count']}"

def analyze_carrier_tree(tree):
    run_history_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_history':
            run_history_node = node
            break
            
    names = [node.id for node in ast.walk(run_history_node) if isinstance(node, ast.Name)] if run_history_node else []
    has_h = any(n == 'h' or n.startswith('h_') for n in names)
    has_spike = any(n in ('fired', 'spike', 'reset') for n in names)
    
    binops = [node for node in ast.walk(run_history_node) if isinstance(node, ast.BinOp)] if run_history_node else []
    has_arith = any(isinstance(op.op, (ast.Add, ast.Sub)) for op in binops)
    
    shortcut_dict_kwargs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'shortcut':
                    if isinstance(node.value, ast.Call):
                        for kw in node.value.keywords:
                            val = kw.value.value if isinstance(kw.value, ast.Constant) else 'EXPRESSION'
                            shortcut_dict_kwargs[kw.arg] = val
                            
    hardcoded_true_count_shortcut = sum(1 for v in shortcut_dict_kwargs.values() if v is True)
    
    direct_check_true_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'check':
            if len(node.args) >= 2:
                arg2 = node.args[1]
                if isinstance(arg2, ast.Constant) and arg2.value is True:
                    direct_check_true_calls.append(node.args[0].value if isinstance(node.args[0], ast.Constant) else str(node.args[0]))
                    
    direct_check_true_count = len(direct_check_true_calls)
    
    return {
        'run_history_analysis': {
            'found': run_history_node is not None,
            'has_membrane_potential_h': has_h,
            'has_spike_or_reset': has_spike,
            'direct_arithmetic_found': has_arith
        },
        'ast_shortcut_kwargs': shortcut_dict_kwargs,
        'direct_check_true_calls': direct_check_true_calls,
        'answers': {
            'hardcoded_true_count_shortcut': hardcoded_true_count_shortcut,
            'direct_check_true_count': direct_check_true_count,
            'q1_depends_on_membrane_update': False,
            'q2_passes_through_spike_reset': False,
            'q3_is_static_retriever_plus_arithmetic': True,
            'q4_hardcoded_true_checks_count': hardcoded_true_count_shortcut + direct_check_true_count,
            'q5_nature_of_proof': 'Operator-level constructive sufficiency under controlled conditions; NOT online neural implementation, NOT learned generalization.'
        }
    }

def analyze_carrier(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    results = analyze_carrier_tree(tree)
    
    print("Composition Carrier Analysis Results:")
    print("  1. Membrane potential (h) in run_history:      ", results['run_history_analysis']['has_membrane_potential_h'])
    print("  2. Spike / reset in run_history:               ", results['run_history_analysis']['has_spike_or_reset'])
    print("  3. Direct explicit arithmetic in run_history:  ", results['run_history_analysis']['direct_arithmetic_found'])
    print("  4. AST shortcut dict kwargs:                   ", results['ast_shortcut_kwargs'])
    print(f"     -> Shortcut dict True count:                {results['answers']['hardcoded_true_count_shortcut']} (5 expected)")
    print(f"     -> Direct check(..., True) calls:           {results['direct_check_true_calls']}")
    print(f"     -> Total structural True checks:            {results['answers']['q4_hardcoded_true_checks_count']}")
    print("  5. Conclusion:                                 ", results['answers']['q5_nature_of_proof'])
    
    verify_carrier_assertions(results)
    print("ALL CARRIER ASSERTIONS PASSED!")
    return results

def test_carrier_error_fixture():
    """Negative testing: mutates code AST inputs and verifies that the actual checker functions REJECT them."""
    print("Running Carrier Error Fixture Tests (testing that real checker functions reject perturbed inputs)...")
    
    # 1. Fake AST with membrane update -> verify_carrier_assertions MUST reject
    fake_code_with_h = '''
def run_history(x):
    h = 0.5 * h + x
    c = a + b
    return h
shortcut = dict(a=True, b=True, c=True, d=True, e=True)
check("test", True)
'''
    tree1 = ast.parse(fake_code_with_h)
    res1 = analyze_carrier_tree(tree1)
    assert res1['run_history_analysis']['has_membrane_potential_h'] is True
    try:
        verify_carrier_assertions(res1)
        raise RuntimeError("verify_carrier_assertions failed to reject code with membrane potential!")
    except AssertionError:
        print("  [PASS] Membrane potential in run_history correctly rejected by verify_carrier_assertions")
        
    # 2. Fake AST with missing shortcut True flags -> verify_carrier_assertions MUST reject
    fake_code_missing_true = '''
def run_history(x):
    return a + b
shortcut = dict(a=False, b=False)
check("test", True)
'''
    tree2 = ast.parse(fake_code_missing_true)
    res2 = analyze_carrier_tree(tree2)
    assert res2['answers']['hardcoded_true_count_shortcut'] == 0
    try:
        verify_carrier_assertions(res2)
        raise RuntimeError("verify_carrier_assertions failed to reject missing True flags!")
    except AssertionError:
        print("  [PASS] Missing shortcut True flags correctly rejected by verify_carrier_assertions")
        
    # 3. Fake AST without arithmetic in run_history -> verify_carrier_assertions MUST reject
    fake_code_no_arith = '''
def run_history(x):
    return x
shortcut = dict(a=True, b=True, c=True, d=True, e=True)
check("test", True)
'''
    tree3 = ast.parse(fake_code_no_arith)
    res3 = analyze_carrier_tree(tree3)
    assert res3['run_history_analysis']['direct_arithmetic_found'] is False
    try:
        verify_carrier_assertions(res3)
        raise RuntimeError("verify_carrier_assertions failed to reject missing arithmetic!")
    except AssertionError:
        print("  [PASS] Missing direct arithmetic in run_history correctly rejected by verify_carrier_assertions")
        
    print("ALL CARRIER ERROR FIXTURES PASSED (Real checker functions verified to reject corrupted data)!")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'snapshot', 'code', 'experiments', 'sprint4_3', 'sprint4_3a_composition_probe.py')
    analyze_carrier(p)
    test_carrier_error_fixture()
