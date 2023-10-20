configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GEMMINI_OPT_TEST_DEPENDS
        FileCheck count not
        gemmini-opt
        )

add_lit_testsuite(check-gemmini-opt "Running the gemmini-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GEMMINI_OPT_TEST_DEPENDS}
        )
set_target_properties(check-gemmini-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(GEMMINI_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GEMMINI_OPT_TEST_DEPENDS})
