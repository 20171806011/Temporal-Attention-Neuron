"""Scalar recurrence transcribed from pinned models_p1b_v2.py:euler_step.
Source read via GitHub connector; this is NOT a downloaded complete source file.
Original MIT copyright Li Zexu, 2026 (see LICENSE). Constants and operation order
below match the inspected recurrence. A scalar schedule is only a local test harness.
"""
def step(u,A,r,I_content,novelty,I_query):
    h=0.1; tau_A=15.; tau_m=10.; alpha=1.; A_max=5.;
    g_A=1.; u_rest=0.; I_bias=0.; theta_th=1.; u_reset=0.; tau_ref=2.
    dA = (-A + alpha * novelty) / tau_A * h
    A_next = min(A_max, max(0.0, A + dA))
    if r > 0.0:
        r_next = max(0.0, r-h)
        u_next = u_reset
        spike=0
        u_cand=None
    else:
        I_syn=g_A*A*I_content
        du=(-(u-u_rest)+I_syn+I_query+I_bias)/tau_m*h
        u_cand=u+du
        if u_cand>=theta_th:
            spike=1;u_next=u_reset;r_next=tau_ref
        else:
            spike=0;u_next=u_cand;r_next=0.0
    return u_next,A_next,r_next,spike,u_cand
