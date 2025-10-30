<?php

/**
 * Add values to global variables
 *
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.3.1
 */

if (!function_exists('widgetopts_register_globals')) {
    add_action('init', 'widgetopts_register_globals', 90);
    function widgetopts_register_globals()
    {
        global $widgetopts_taxonomies, $widgetopts_types, $widgetopts_categories;

        $widgetopts_taxonomies     = widgetopts_global_taxonomies();
        $widgetopts_types         = widgetopts_global_types();
        $widgetopts_categories     = widgetopts_global_categories();
    }
}

if (!function_exists('widgetopts_removed_widget_cached')) {
    add_action('admin_init', 'widgetopts_removed_widget_cached', 90);
    function widgetopts_removed_widget_cached()
    {
        $cached = get_option('widgetopts_editor_cached');
        if ($cached) {
            $_cached = json_decode($cached, true);
            if (isset($_cached) && !empty($_cached)) {
                $_cached = (array) $_cached;
                if (is_iterable($_cached)) {
                    foreach ($_cached as $key => $c) {
                        if (!empty($c['widgetopts_expiry'])) {
                            if (time() > strtotime($c['widgetopts_expiry'])) {
                                unset($_cached[$key]);
                            }
                        }
                    }

                    update_option('widgetopts_editor_cached', json_encode($_cached));
                }
            }
        }
    }
}
