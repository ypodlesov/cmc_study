#include <vector>
#include <map>
#include <functional>

namespace numbers {
complex eval(const std::vector<std::string> &args, const complex &z) {
    std::map<char, std::function<complex_stack(complex_stack, complex, 
                                 std::string)>> func{};
    func['('] = [](complex_stack st, complex z, std::string s) { 
        return st << complex(s);
    };
    func['z'] = [](complex_stack st, complex z, std::string s) { 
        return st << z;
    };
    func['+'] = [](complex_stack st, complex z, std::string s) { 
        return (~(~st)) << (+st + +(~st));
    };
    func['-'] = [](complex_stack st, complex z, std::string s) { 
        return (~(~st)) << (+(~st) - +st);
    };
    func['*'] = [](complex_stack st, complex z, std::string s) { 
        return (~(~st)) << (+(~st) * +st);
    };
    func['/'] = [](complex_stack st, complex z, std::string s) { 
        return (~(~st)) << (+(~st) / +st);
    };
    func['!'] = [](complex_stack st, complex z, std::string s) { 
        return st << +st;
    };
    func[';'] = [](complex_stack st, complex z, std::string s) { 
        return ~st;
    };
    func['~'] = [](complex_stack st, complex z, std::string s) { 
        return ~st << ~(+st);
    };
    func['#'] = [](complex_stack st, complex z, std::string s) { 
        return ~st << -(+st);
    };
    complex_stack st{};
    for (const auto &s : args) {
        st = func[*s.begin()](st, z, s);
    }
    return +st;
}
}
