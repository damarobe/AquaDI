def decode_hash_to_params(binary_hash, param_ranges):
    params = []
    hash_chunks = torch.chunk(binary_hash, len(param_ranges), dim=1)
    for chunk, (min_val, max_val) in zip(hash_chunks, param_ranges):
        int_val = chunk.float().matmul(torch.pow(2, torch.arange(chunk.shape[1] - 1, -1, -1).float()))
        decoded = min_val + (max_val - min_val) * int_val / (2 ** chunk.shape[1] - 1)
        params.append(decoded)
    return torch.cat(params, dim=1)
