#pragma once

#if defined(_MSC_VER)

namespace std 
{
  // Manually implement a signbit function as defined in the standard.
  // It is non-trivial, since it must return valid values for +/-0, +/-inf
  // as well, so the best thing really is to use the bit representation of the
  // floating point numbers. Hence, works only for IEEE floats!
  template <typename T>
  inline static bool signbit(const T &x);
  template <>
  inline static bool signbit<float>(const float &x) {
    return (*reinterpret_cast<const long*>(&x) & (1L << 31)) != 0;
  }
  template <>
  inline static bool signbit<double>(const double &x) {
    return (*reinterpret_cast<const long long*>(&x) & (1LL << 63)) != 0;
  }
}

#if _MSC_VER < 1800
#include <float.h>
#include <math.h>

namespace std {
  template <typename T>
  bool isnan(const T &x) { return _isnan(x); }

  template <typename T>
  bool isinf(const T &x) { return !_finite(x); }
}

template <typename T>
inline T round(T number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
/*
inline double log2( double n )
{
    return log( n ) / log( 2.0 );
}*/

#endif

#endif