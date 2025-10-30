<?php

/**
 * Handle compatibility with Pagebuilder by SiteOrigin Plugin
 *
 * Process AJAX actions.
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */
// Exit if accessed directly
if (! defined('ABSPATH')) exit;

if (!function_exists('widgetopts_siteorigin_panels_data')) {
    add_filter('siteorigin_panels_data', 'widgetopts_siteorigin_panels_data', 10, 4);
    function widgetopts_siteorigin_panels_data($panels_data, $post_id)
    {
        global $widget_options;
        if (!is_admin()) {
            if (isset($panels_data['widgets']) && !empty($panels_data['widgets']) && is_array($panels_data['widgets'])) {

                global $current_user;

                foreach ($panels_data['widgets'] as $key => $widgets) {
                    if (isset($widgets['extended_widget_opts']) && !empty($widgets['extended_widget_opts'])) {

                        if (isset($panels_data['widgets'][$key]) && 'activate' == $widget_options['logic']) {
                            // display widget logic
                            if (isset($widgets['extended_widget_opts']['class']) && isset($widgets['extended_widget_opts']['class']['logic']) && !empty($widgets['extended_widget_opts']['class']['logic'])) {
                                $display_logic = stripslashes(trim($widgets['extended_widget_opts']['class']['logic']));
                                $display_logic = apply_filters('widget_options_logic_override', $display_logic);
                                if ($display_logic === false) {
                                    unset($panels_data['widgets'][$key]);
                                }
                                if ($display_logic === true) {
                                    // return true;
                                }
                                // if (stristr($display_logic, "return") === false) {
                                //     $display_logic = "return (" . $display_logic . ");";
                                // }
                                $display_logic = htmlspecialchars_decode($display_logic, ENT_QUOTES);
                                try {
                                    if (!widgetopts_safe_eval($display_logic)) {
                                        unset($panels_data['widgets'][$key]);
                                    }
                                } catch (ParseError $e) {
                                    unset($panels_data['widgets'][$key]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return $panels_data;
    }
}

if (!function_exists('widgetopts_siteorigin_panels_widget_classes')) {
    add_filter('siteorigin_panels_widget_classes', 'widgetopts_siteorigin_panels_widget_classes', 10, 4);
    function widgetopts_siteorigin_panels_widget_classes($classes, $widget, $instance, $widget_info)
    {
        if (isset($instance['extended_widget_opts'])) {
            global $widget_options;

            $get_classes    = widgetopts_classes_generator($instance['extended_widget_opts'], $widget_options, $widget_options['settings'], true);
            $get_classes[]  = 'widgetopts-SO';

            $classes = apply_filters('widgetopts_siteorigin_panels_widget_classes', array_merge($classes, $get_classes), $widget_info);
        }

        return $classes;
    }
}
