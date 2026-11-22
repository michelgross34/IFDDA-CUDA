#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>

#ifdef USE_SIMP
using FortranReal = float;
#else
using FortranReal = double;
#endif

struct FortranComplex {
    FortranReal real;
    FortranReal imag;
};

//-----------------------------------------------------------
extern "C" void calculatedate_(const int valuesf[8], const int valuesi[8],
                                const FortranReal* tf, const FortranReal* ti,
                                const char* message, FortranReal* elapsed_real,
                                std::size_t message_length) {
    if (valuesf == nullptr || valuesi == nullptr || tf == nullptr || ti == nullptr ||
        message == nullptr || elapsed_real == nullptr) {
        return;
    }

    std::size_t message_size = message_length;
    while (message_size > 0 && message[message_size - 1] == ' ') {
        --message_size;
    }

    const FortranReal elapsed_cpu = *tf - *ti;
    std::cout << " CPU time " << std::string(message, message_size)
              << " in second  : " << elapsed_cpu << '\n';

    FortranReal elapsed_days = FortranReal(0);
    if (valuesf[2] > valuesi[2]) {
        elapsed_days = FortranReal(valuesf[2] - valuesi[2]) * FortranReal(86400);
    } else if (valuesf[2] < valuesi[2]) {
        static constexpr int month_days[12] = {
            31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        const int start_month = valuesi[1];
        if (start_month >= 1 && start_month <= 12) {
            elapsed_days = FortranReal(valuesf[2] + month_days[start_month - 1] -
                                       valuesi[2]) * FortranReal(86400);
        }
    }

    const FortranReal elapsed_clock =
        FortranReal(valuesf[4] - valuesi[4]) * FortranReal(3600) +
        FortranReal(valuesf[5] - valuesi[5]) * FortranReal(60) +
        FortranReal(valuesf[6] - valuesi[6]) +
        FortranReal(valuesf[7] - valuesi[7]) / FortranReal(1000);
    *elapsed_real = elapsed_clock + elapsed_days;

    std::cout << " Real time " << std::string(message, message_size)
              << " in second : " << *elapsed_real << '\n';
}

//-----------------------------------------------------------
extern "C" void factorto235_(int* value) {
    if (value == nullptr) {
        return;
    }

    std::cout << " old N " << *value << '\n';
    int candidate = *value;
    while (candidate <= std::numeric_limits<int>::max()) {
        int remainder = candidate;
        if (remainder > 1) {
            for (const int factor : {2, 3, 5}) {
                while (remainder % factor == 0) {
                    remainder /= factor;
                }
            }
            if (remainder == 1) {
                *value = candidate;
                std::cout << " new N " << *value << '\n';
                return;
            }
        }
        if (candidate == std::numeric_limits<int>::max()) {
            return;
        }
        ++candidate;
    }
}

//-----------------------------------------------------------
extern "C" void primefactor_(const int* value, const int* component, int* test) {
    if (value == nullptr || component == nullptr || test == nullptr) {
        return;
    }

    int remainder = *value;
    int counts[9] = {};
    *test = 0;
    const int fast_factors[3] = {2, 3, 5};
    const int remaining_factors[6] = {7, 11, 13, 17, 19, 23};
    for (int i = 0; i < 3 && remainder > 1; ++i) {
        while (remainder % fast_factors[i] == 0) {
            ++counts[i];
            remainder /= fast_factors[i];
        }
    }
    if (remainder > 1) {
        for (int i = 0; i < 6 && remainder > 1; ++i) {
            while (remainder % remaining_factors[i] == 0) {
                ++counts[i + 3];
                remainder /= remaining_factors[i];
            }
        }
    }
    if (remainder > 1) {
        std::cout << " High prime factor: FFT can be slowed down\n";
        if (*component == 1) {
            std::cout << " for the discretization along the x axis\n";
        } else if (*component == 2) {
            std::cout << " for the discretization along the y axis\n";
        } else {
            std::cout << " for the discretization along the z axis\n";
        }
        std::cout << "2^" << counts[0] << "*3^" << counts[1] << "*5^" << counts[2]
                  << "*7^" << counts[3] << "*11^" << counts[4] << "*13^" << counts[5]
                  << "*17^" << counts[6] << "*19^" << counts[7] << "*23^" << counts[8]
                  << "*N=" << remainder << '\n';
    }
}

//-----------------------------------------------------------
extern "C" void calculangle_(FortranReal* theta, FortranReal* phi,
                              const FortranReal* k0, const FortranReal* cell_size,
                              const int* fft_size) {
    if (theta == nullptr || phi == nullptr || k0 == nullptr || cell_size == nullptr ||
        fft_size == nullptr || *fft_size == 0 || *cell_size == FortranReal(0) ||
        *k0 == FortranReal(0) || *theta == FortranReal(0)) {
        return;
    }

    constexpr FortranReal pi =
        static_cast<FortranReal>(3.141592653589793238462643383279502884L);
    std::cout << " old angle " << *theta << ' ' << *phi << '\n';

    const FortranReal delta_k = FortranReal(2) * pi /
                                (*cell_size * static_cast<FortranReal>(*fft_size));
    const FortranReal theta_rad = *theta * pi / FortranReal(180);
    const FortranReal phi_rad = *phi * pi / FortranReal(180);
    const FortranReal kx_incident = std::sin(theta_rad) * std::cos(phi_rad) * *k0;
    const FortranReal ky_incident = std::sin(theta_rad) * std::sin(phi_rad) * *k0;
    const int ix = static_cast<int>(std::round(kx_incident / delta_k));
    const int iy = static_cast<int>(std::round(ky_incident / delta_k));
    const FortranReal kx = delta_k * static_cast<FortranReal>(ix);
    const FortranReal ky = delta_k * static_cast<FortranReal>(iy);
    const FortranReal k_parallel = std::sqrt(kx * kx + ky * ky);
    const FortranReal new_theta_rad = std::asin(k_parallel / *k0);
    FortranReal new_phi_rad = std::atan2(ky, kx);

    *theta = std::copysign(new_theta_rad * FortranReal(180) / pi, *theta);
    if (*theta < FortranReal(0)) {
        new_phi_rad += pi;
    }
    *phi = new_phi_rad * FortranReal(180) / pi;
    std::cout << " new angle to fit mesh of FFT " << *theta << ' ' << *phi << '\n';
}

//-----------------------------------------------------------
extern "C" void writecolor_(const char* data, const char* color,
                             std::size_t data_length, std::size_t color_length) {
    if (data == nullptr || color == nullptr) {
        return;
    }

    std::size_t data_size = data_length;
    while (data_size > 0 && data[data_size - 1] == ' ') {
        --data_size;
    }
    const std::string text(data, data_size);
    const std::string color_name(color, color_length);
    const auto matches = [&color_name](const char* prefix) {
        const std::string key(prefix);
        return color_name.compare(0, key.size(), key) == 0;
    };
    const auto print = [&text](const char* ansi) {
        std::cout << ' ' << '\x1b' << '[' << ansi << 'm' << text << '\x1b'
                  << "[0m\n";
    };

    if (matches("boldface")) print("1");
    if (matches("italic")) print("3");
    if (matches("underline")) print("4");
    if (matches("blink")) print("5");
    if (matches("bgblack")) print("7");
    if (matches("cross")) print("9");
    if (matches("reddark")) print("31");
    if (matches("greendark")) print("32");
    if (matches("brown")) print("33");
    if (matches("bluedark")) print("34");
    if (matches("violet")) print("35");
    if (matches("cyandark")) print("36");
    if (matches("bgreddark")) print("41");
    if (matches("bggreendark")) print("42");
    if (matches("bgbrown")) print("43");
    if (matches("bgbluedark")) print("44");
    if (matches("bgviolet")) print("45");
    if (matches("bgcyandark")) print("46");
    if (matches("red")) print("91");
    if (matches("green")) print("92");
    if (matches("yellow")) print("93");
    if (matches("blue")) print("94");
    if (matches("pink")) print("95");
    if (matches("cyan")) print("96");
    if (matches("bggray")) print("100");
    if (matches("bgred")) print("101");
    if (matches("bggreen")) print("102");
    if (matches("bgyellow")) print("103");
    if (matches("bgblue")) print("104");
    if (matches("bgpink")) print("105");
    if (matches("bgcyan")) print("106");
}

//-----------------------------------------------------------
extern "C" void irradiance_(const FortranReal* power, const FortranReal* waist,
                             FortranComplex* field, FortranReal* irradiance) {
    if (power == nullptr || waist == nullptr || field == nullptr || irradiance == nullptr ||
        *waist == FortranReal(0)) {
        return;
    }

    constexpr FortranReal speed_of_light = FortranReal(299792458);
    constexpr FortranReal pi =
        static_cast<FortranReal>(3.141592653589793238462643383279502884L);
    const FortranReal epsilon_0 =
        FortranReal(1) / (speed_of_light * speed_of_light * FortranReal(4) * pi *
                          FortranReal(1.0e-7));
    const FortranReal surface = *waist * *waist * pi;
    *irradiance = *power / surface;
    field->real = std::sqrt(FortranReal(2) * *irradiance / speed_of_light / epsilon_0);
    field->imag = FortranReal(0);
    std::cout << ' ' << epsilon_0 << ' ' << *power << ' ' << *waist << ' ' << surface
              << ' ' << *irradiance << ' ' << '(' << field->real << ',' << field->imag
              << ")\n";
}

//-----------------------------------------------------------
extern "C" void copiani_(const FortranComplex* epsilon, FortranComplex* tensor,
                          const int* maximum_dipoles, const int* isotropy,
                          const int* dipole_index) {
    if (epsilon == nullptr || tensor == nullptr || maximum_dipoles == nullptr ||
        isotropy == nullptr || dipole_index == nullptr || *maximum_dipoles <= 0 ||
        *dipole_index < 1 || *dipole_index > *maximum_dipoles ||
        (*isotropy != 1 && *isotropy != 3)) {
        return;
    }

    if (*isotropy == 1) {
        const FortranComplex scalar = epsilon[*dipole_index - 1];
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row) {
                const size_t output_index = size_t(row) + size_t(column) * 3;
                tensor[output_index] = row == column ? scalar : FortranComplex{0, 0};
            }
        }
        return;
    }

    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            const size_t input_index = size_t(*dipole_index - 1) +
                                       size_t(row) * size_t(*maximum_dipoles) +
                                       size_t(column) * size_t(*maximum_dipoles) * 3;
            const size_t output_index = size_t(row) + size_t(column) * 3;
            tensor[output_index] = epsilon[input_index];
        }
    }
}

//-----------------------------------------------------------
extern "C" void sldrr_(const FortranReal* ss_input, const FortranReal* pp_input,
                        FortranReal* theta_degrees, FortranReal* phi_degrees,
                        FortranReal* sldr, int* stop, char* info,
                        std::size_t info_length) {
    if (ss_input == nullptr || pp_input == nullptr || theta_degrees == nullptr ||
        phi_degrees == nullptr || sldr == nullptr || stop == nullptr || info == nullptr) {
        return;
    }
    std::fill(info, info + info_length, ' ');
    *stop = 0;

    constexpr FortranReal pi =
        static_cast<FortranReal>(3.141592653589793238462643383279502884L);
    const FortranReal theta = *theta_degrees * pi / FortranReal(180);
    const FortranReal phi = *phi_degrees * pi / FortranReal(180);
    FortranReal pp = *pp_input;
    const FortranReal ss = *ss_input;
    FortranReal p = 0;
    FortranReal s = 0;
    const auto matches = [](FortranReal value, FortranReal reference) {
        if (reference == FortranReal(0)) {
            return value == FortranReal(0);
        }
        return std::abs((value - reference) / reference) <= FortranReal(1e-12);
    };

    const auto set_orientation = [&p, &s, &pp](FortranReal phi_degrees) {
        p = std::sqrt(pp);
        s = std::sqrt(FortranReal(1) - pp);
        if (phi_degrees >= FortranReal(90) && phi_degrees < FortranReal(180)) {
            s = -s;
        } else if (phi_degrees >= FortranReal(180) && phi_degrees < FortranReal(270)) {
            p = -p;
            s = -s;
        } else if (phi_degrees >= FortranReal(270) && phi_degrees < FortranReal(360)) {
            p = -p;
        }
    };

    FortranReal normalized_phi = *phi_degrees;
    if (matches(pp, FortranReal(3))) {
        if (normalized_phi < FortranReal(0)) {
            normalized_phi += FortranReal(360);
        }
        const FortranReal sin_phi = std::sin(phi);
        const FortranReal cos_theta = std::cos(theta);
        const FortranReal cos_phi = std::cos(phi);
        pp = (sin_phi * sin_phi) /
             (cos_theta * cos_theta * cos_phi * cos_phi + sin_phi * sin_phi);
        if (normalized_phi >= FortranReal(0) && normalized_phi < FortranReal(360)) {
            set_orientation(normalized_phi);
        }
    }
    if (matches(pp, FortranReal(2))) {
        pp = std::pow(std::cos(phi), FortranReal(2)) /
             (std::pow(std::cos(theta), FortranReal(2)) * std::pow(std::sin(phi), FortranReal(2)) +
              std::pow(std::cos(phi), FortranReal(2)));
        if (normalized_phi < FortranReal(0)) {
            normalized_phi += FortranReal(360);
        }
        if (normalized_phi >= FortranReal(0) && normalized_phi < FortranReal(360)) {
            set_orientation(normalized_phi);
        }
    }
    *phi_degrees = normalized_phi;
    if (pp < FortranReal(0) || pp > FortranReal(1)) {
        *stop = -1;
        const char message[] = "problem in plane incident wave";
        std::copy(message, message + sizeof(message) - 1, info);
        return;
    }

    const FortranReal direction_x = std::sin(theta) * std::cos(phi);
    const FortranReal direction_y = std::sin(theta) * std::sin(phi);
    const FortranReal direction_z = std::cos(theta);
    const FortranReal field_x = -s * std::sin(phi) + p * std::cos(theta) * std::cos(phi);
    const FortranReal field_y = s * std::cos(phi) + p * std::cos(theta) * std::sin(phi);
    const FortranReal field_z = -p * std::sin(theta);
    const FortranReal product_x = direction_x * field_x;
    const FortranReal product_y = direction_y * field_y;
    const FortranReal product_z = direction_z * field_z;
    *sldr = product_x * product_x + product_y * product_y + product_z * product_z;
    std::cout << " SLDR=" << *sldr << " verif="
              << product_x + product_y + product_z << '\n';
}
