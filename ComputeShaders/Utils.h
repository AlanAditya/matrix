//
//  Utils.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 14/07/25.
//

#ifndef Utils_h
#define Utils_h

#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#endif /* Utils_h */
