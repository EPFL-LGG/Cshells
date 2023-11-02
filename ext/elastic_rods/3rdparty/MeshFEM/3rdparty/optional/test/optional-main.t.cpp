// Copyright (c) 2016 Martin Moene
//
// https://github.com/martinmoene/optional-lite
//
// Distributed under the Boost Software License, Version 1.0. 
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "optional-main.t.h"

#ifndef  optional_HAVE
# define optional_HAVE(FEATURE) ( optional_HAVE_##FEATURE )
#endif

#define optional_PRESENT( x ) \
    std::cout << #x << ": " << x << "\n"

#define optional_ABSENT( x ) \
    std::cout << #x << ": (undefined)\n"

lest::tests & specification()
{
    static lest::tests tests;
    return tests;
}

CASE( "__cplusplus" "[.stdc++]" )
{
    optional_PRESENT( __cplusplus );
}

CASE( "optional-lite version" "[.version]" )
{
    optional_PRESENT( optional_lite_VERSION );
}

CASE( "compiler version" "[.compiler]" )
{
#ifdef optional_COMPILER_CLANG_VERSION
    optional_PRESENT( optional_COMPILER_CLANG_VERSION );
#else
    optional_ABSENT(  optional_COMPILER_CLANG_VERSION );
#endif

#ifdef optional_COMPILER_GNUC_VERSION
    optional_PRESENT( optional_COMPILER_GNUC_VERSION );
#else
    optional_ABSENT(  optional_COMPILER_GNUC_VERSION );
#endif

#ifdef optional_COMPILER_MSVC_VERSION
    optional_PRESENT( optional_COMPILER_MSVC_VERSION );
#else
    optional_ABSENT(  optional_COMPILER_MSVC_VERSION );
#endif
}

CASE( "presence of C++ language features" "[.stdlanguage]" )
{
#if optional_HAVE( AUTO )
    optional_PRESENT( optional_HAVE_AUTO );
#else
    optional_ABSENT(  optional_HAVE_AUTO );
#endif

#if optional_HAVE( NULLPTR )
    optional_PRESENT( optional_HAVE_NULLPTR );
#else
    optional_ABSENT(  optional_HAVE_NULLPTR );
#endif

#if optional_HAVE( STATIC_ASSERT )
    optional_PRESENT( optional_HAVE_STATIC_ASSERT );
#else
    optional_ABSENT(  optional_HAVE_STATIC_ASSERT );
#endif

#if optional_HAVE( DEFAULT_FUNCTION_TEMPLATE_ARG )
    optional_PRESENT( optional_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG );
#else
    optional_ABSENT(  optional_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG );
#endif

#if optional_HAVE( ALIAS_TEMPLATE )
    optional_PRESENT( optional_HAVE_ALIAS_TEMPLATE );
#else
    optional_ABSENT(  optional_HAVE_ALIAS_TEMPLATE );
#endif

#if optional_HAVE( CONSTEXPR_11)
    optional_PRESENT( optional_HAVE_CONSTEXPR_11 );
#else
    optional_ABSENT(  optional_HAVE_CONSTEXPR_11 );
#endif

#if optional_HAVE( CONSTEXPR_14 )
    optional_PRESENT( optional_HAVE_CONSTEXPR_14 );
#else
    optional_ABSENT(  optional_HAVE_CONSTEXPR_14 );
#endif

#if optional_HAVE( ENUM_CLASS )
    optional_PRESENT( optional_HAVE_ENUM_CLASS );
#else
    optional_ABSENT(  optional_HAVE_ENUM_CLASS );
#endif

#if optional_HAVE( ENUM_CLASS_CONSTRUCTION_FROM_UNDERLYING_TYPE )
    optional_PRESENT( optional_HAVE_ENUM_CLASS_CONSTRUCTION_FROM_UNDERLYING_TYPE );
#else
    optional_ABSENT(  optional_HAVE_ENUM_CLASS_CONSTRUCTION_FROM_UNDERLYING_TYPE );
#endif

#if optional_HAVE( EXPLICIT_CONVERSION )
    optional_PRESENT( optional_HAVE_EXPLICIT_CONVERSION );
#else
    optional_ABSENT(  optional_HAVE_EXPLICIT_CONVERSION );
#endif

#if optional_HAVE( INITIALIZER_LIST )
    optional_PRESENT( optional_HAVE_INITIALIZER_LIST );
#else
    optional_ABSENT(  optional_HAVE_INITIALIZER_LIST );
#endif

#if optional_HAVE( IS_DEFAULT )
    optional_PRESENT( optional_HAVE_IS_DEFAULT );
#else
    optional_ABSENT(  optional_HAVE_IS_DEFAULT );
#endif

#if optional_HAVE( IS_DELETE )
    optional_PRESENT( optional_HAVE_IS_DELETE );
#else
    optional_ABSENT(  optional_HAVE_IS_DELETE );
#endif

#if optional_HAVE( NOEXCEPT )
    optional_PRESENT( optional_HAVE_NOEXCEPT );
#else
    optional_ABSENT(  optional_HAVE_NOEXCEPT );
#endif

#if optional_HAVE( REF_QUALIFIER )
    optional_PRESENT( optional_HAVE_REF_QUALIFIER );
#else
    optional_ABSENT(  optional_HAVE_REF_QUALIFIER );
#endif
}

CASE( "presence of C++ library features" "[.stdlibrary]" )
{
#if optional_HAVE( ARRAY )
    optional_PRESENT( optional_HAVE_ARRAY );
#else
    optional_ABSENT(  optional_HAVE_ARRAY );
#endif

#if optional_HAVE( CONDITIONAL )
    optional_PRESENT( optional_HAVE_CONDITIONAL );
#else
    optional_ABSENT(  optional_HAVE_CONDITIONAL );
#endif

#if optional_HAVE( CONTAINER_DATA_METHOD )
    optional_PRESENT( optional_HAVE_CONTAINER_DATA_METHOD );
#else
    optional_ABSENT(  optional_HAVE_CONTAINER_DATA_METHOD );
#endif

#if optional_HAVE( REMOVE_CV )
    optional_PRESENT( optional_HAVE_REMOVE_CV );
#else
    optional_ABSENT(  optional_HAVE_REMOVE_CV );
#endif

#if optional_HAVE( SIZED_TYPES )
    optional_PRESENT( optional_HAVE_SIZED_TYPES );
#else
    optional_ABSENT(  optional_HAVE_SIZED_TYPES );
#endif

#if optional_HAVE( TYPE_TRAITS )
    optional_PRESENT( optional_HAVE_TYPE_TRAITS );
#else
    optional_ABSENT(  optional_HAVE_TYPE_TRAITS );
#endif

#ifdef _HAS_CPP0X
    optional_PRESENT( _HAS_CPP0X );
#else
    optional_ABSENT(  _HAS_CPP0X );
#endif
}

int main( int argc, char * argv[] )
{
    return lest::run( specification(), argc, argv );
}

#if 0
g++            -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++98 -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++03 -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++0x -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++11 -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++14 -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass
g++ -std=c++17 -I../include/nonstd -o optional-lite.t.exe optional-lite.t.cpp && optional-lite.t.exe --pass

cl -EHsc -I../include/nonstd optional-lite.t.cpp && optional-lite.t.exe --pass
#endif

// end of file
