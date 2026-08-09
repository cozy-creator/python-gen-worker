AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2412_1 = ConstantHandle(input_handles[0]);
    int32_t arg2412_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2412_1, &arg2412_1_dtype));

    int32_t arg2412_1_expected_dtype = aoti_torch_dtype_bfloat16();
    if (arg2412_1_expected_dtype != arg2412_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg2412_1_expected_dtype << "(at::kBFloat16), "
           << "but got: " << arg2412_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2412_1_size = arg2412_1.sizes();

    if (1 != arg2412_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg2412_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (4 != arg2412_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 4, " << "but got: " << arg2412_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (80 != arg2412_1_size[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 2, "
           << "expected: 80, " << "but got: " << arg2412_1_size[2]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (192 != arg2412_1_size[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 3, "
           << "expected: 192, " << "but got: " << arg2412_1_size[3]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2412_1_stride = arg2412_1.strides();

    if (61440 != arg2412_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 61440, " << "but got: " << arg2412_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (15360 != arg2412_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 15360, " << "but got: " << arg2412_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (192 != arg2412_1_stride[2]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 2, "
           << "expected: 192, " << "but got: " << arg2412_1_stride[2]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg2412_1_stride[3]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 3, "
           << "expected: 1, " << "but got: " << arg2412_1_stride[3]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg2412_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2412_1, &arg2412_1_device_type));

    int32_t arg2412_1_expected_device_type = 1;
    if (arg2412_1_expected_device_type != arg2412_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg2412_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2412_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_1(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2413_1 = ConstantHandle(input_handles[1]);
    int32_t arg2413_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2413_1, &arg2413_1_dtype));

    int32_t arg2413_1_expected_dtype = aoti_torch_dtype_bfloat16();
    if (arg2413_1_expected_dtype != arg2413_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dtype, "
           << "expected: " << arg2413_1_expected_dtype << "(at::kBFloat16), "
           << "but got: " << arg2413_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2413_1_size = arg2413_1.sizes();
    auto arg2413_1_stride = arg2413_1.strides();
    int32_t arg2413_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2413_1, &arg2413_1_device_type));

    int32_t arg2413_1_expected_device_type = 1;
    if (arg2413_1_expected_device_type != arg2413_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched device type, "
        << "expected: " << arg2413_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2413_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_2(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2414_1 = ConstantHandle(input_handles[2]);
    int32_t arg2414_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2414_1, &arg2414_1_dtype));

    int32_t arg2414_1_expected_dtype = aoti_torch_dtype_bfloat16();
    if (arg2414_1_expected_dtype != arg2414_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dtype, "
           << "expected: " << arg2414_1_expected_dtype << "(at::kBFloat16), "
           << "but got: " << arg2414_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2414_1_size = arg2414_1.sizes();

    if (1 != arg2414_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg2414_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (77 != arg2414_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dim value at 1, "
           << "expected: 77, " << "but got: " << arg2414_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (2048 != arg2414_1_size[2]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dim value at 2, "
           << "expected: 2048, " << "but got: " << arg2414_1_size[2]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2414_1_stride = arg2414_1.strides();

    if (157696 != arg2414_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched stride value at 0, "
           << "expected: 157696, " << "but got: " << arg2414_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (2048 != arg2414_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched stride value at 1, "
           << "expected: 2048, " << "but got: " << arg2414_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg2414_1_stride[2]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched stride value at 2, "
           << "expected: 1, " << "but got: " << arg2414_1_stride[2]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg2414_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2414_1, &arg2414_1_device_type));

    int32_t arg2414_1_expected_device_type = 1;
    if (arg2414_1_expected_device_type != arg2414_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched device type, "
        << "expected: " << arg2414_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2414_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_3(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2419_1 = ConstantHandle(input_handles[3]);
    int32_t arg2419_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2419_1, &arg2419_1_dtype));

    int32_t arg2419_1_expected_dtype = aoti_torch_dtype_bfloat16();
    if (arg2419_1_expected_dtype != arg2419_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched dtype, "
           << "expected: " << arg2419_1_expected_dtype << "(at::kBFloat16), "
           << "but got: " << arg2419_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2419_1_size = arg2419_1.sizes();

    if (1 != arg2419_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg2419_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1280 != arg2419_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched dim value at 1, "
           << "expected: 1280, " << "but got: " << arg2419_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2419_1_stride = arg2419_1.strides();

    if (1280 != arg2419_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched stride value at 0, "
           << "expected: 1280, " << "but got: " << arg2419_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg2419_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg2419_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg2419_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2419_1, &arg2419_1_device_type));

    int32_t arg2419_1_expected_device_type = 1;
    if (arg2419_1_expected_device_type != arg2419_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[3]: unmatched device type, "
        << "expected: " << arg2419_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2419_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_4(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2420_1 = ConstantHandle(input_handles[4]);
    int32_t arg2420_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2420_1, &arg2420_1_dtype));

    int32_t arg2420_1_expected_dtype = aoti_torch_dtype_bfloat16();
    if (arg2420_1_expected_dtype != arg2420_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched dtype, "
           << "expected: " << arg2420_1_expected_dtype << "(at::kBFloat16), "
           << "but got: " << arg2420_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2420_1_size = arg2420_1.sizes();

    if (1 != arg2420_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg2420_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (6 != arg2420_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched dim value at 1, "
           << "expected: 6, " << "but got: " << arg2420_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2420_1_stride = arg2420_1.strides();

    if (6 != arg2420_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched stride value at 0, "
           << "expected: 6, " << "but got: " << arg2420_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg2420_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg2420_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg2420_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg2420_1, &arg2420_1_device_type));

    int32_t arg2420_1_expected_device_type = 1;
    if (arg2420_1_expected_device_type != arg2420_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[4]: unmatched device type, "
        << "expected: " << arg2420_1_expected_device_type << "1(cuda), "
        << "but got: " << arg2420_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

