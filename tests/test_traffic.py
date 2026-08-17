from request.traffic import TrafficGenerator


def test_deterministic_under_seed(base_config, topology):
    gen_a = TrafficGenerator(base_config.traffic, topology)
    gen_b = TrafficGenerator(base_config.traffic, topology)
    for _ in range(20):
        gen_a.advance_time()
        gen_b.advance_time()
        ra, rb = gen_a.generate_request(), gen_b.generate_request()
        assert (ra.source, ra.destination, ra.b_cc, ra.b_dc) == (rb.source, rb.destination, rb.b_cc, rb.b_dc)


def test_request_fields_are_consistent(base_config, topology):
    gen = TrafficGenerator(base_config.traffic, topology)
    gen.advance_time()
    req = gen.generate_request()
    assert req.source != req.destination
    assert req.departure_time > req.arrival_time
    assert req.update_time == req.arrival_time + base_config.traffic.key_update_period
    assert req.b_qc == base_config.traffic.b_qc


def test_time_is_monotonic(base_config, topology):
    gen = TrafficGenerator(base_config.traffic, topology)
    times = [gen.advance_time() for _ in range(10)]
    assert times == sorted(times)


def test_reset_reproduces_stream(base_config, topology):
    gen = TrafficGenerator(base_config.traffic, topology)
    gen.advance_time()
    first = gen.generate_request()
    gen.reset()
    gen.advance_time()
    again = gen.generate_request()
    assert (first.source, first.destination) == (again.source, again.destination)
