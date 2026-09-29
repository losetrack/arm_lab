from vision_arm_lab.evaluation import summarize_results


def test_summary_counts_all_attempts_and_only_success_times():
    results = [
        {'status': 'success', 'sim_time_s': 2},
        {'status': 'success', 'sim_time_s': 4},
        {'status': 'policy_failure', 'failure_reason': 'success', 'sim_time_s': 0},
        {'status': 'environment_error', 'sim_time_s': None},
    ]
    assert summarize_results(results) == {
        'episodes': 4, 'successes': 2, 'success_rate': 0.5,
        'mean_success_time_s': 3,
        'outcomes': {'success': 2, 'policy_failure': 1, 'environment_error': 1},
    }
    empty = summarize_results([])
    assert empty['episodes'] == empty['success_rate'] == 0
    assert empty['mean_success_time_s'] is None
