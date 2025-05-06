<?php

/**
 * Handles Front-end Display
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Handles widget_display_callback filter
 *
 * @since 1.0
 * @global $widget_options
 * @return $instance
 */

//check if function exists
if (!function_exists('widgetopts_display_callback')) :
    function widgetopts_display_callback($instance, $widget, $args)
    {
        global $widget_options, $current_user, $pagenow, $wp_customize;

        if (empty($instance['extended_widget_opts-' . $widget->id]) && isset($instance['content']) && !empty($instance['content'])) {
            $block = parse_blocks($instance['content']);
            if (!empty($block[0]) && !empty($block[0]['attrs']) && !empty($block[0]['attrs']['extended_widget_opts'])) {
                $instance['extended_widget_opts-' . $widget->id] = $block[0]['attrs']['extended_widget_opts'];
            }
        }

        if (isset($wp_customize)) {
        } else if ($pagenow === 'widgets.php' || (defined('REST_REQUEST') && REST_REQUEST)) {
            return $instance;
        }

        // WPML FIX
        $hasWPML = has_filter('wpml_current_language');
        $hasWPML = (function_exists('pll_the_languages')) ? false : $hasWPML;
        $default_language = $hasWPML ? apply_filters('wpml_default_language', NULL) : false;

        $hidden     = false;
        $opts       = (isset($instance['extended_widget_opts-' . $widget->id])) ? $instance['extended_widget_opts-' . $widget->id] : array();
        $visibility = array('show' => array(), 'hide' => array());

        //wordpress pages
        $visibility         = isset($opts['visibility']) ? $opts['visibility'] : array();
        $visibility_opts    = isset($opts['visibility']['options']) ? $opts['visibility']['options'] : 'hide';

        $is_misc    = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc'])) ? true : false;
        $is_types   = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type'])) ? true : false;
        $is_tax     = ('activate' == $widget_options['visibility'] && isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies'])) ? true : false;

        //check device
        if ($widget_options['devices'] == 'activate' && isset($opts['devices']) && !empty($opts['devices'])) {

            //for mobile and tablet
            if (wp_is_mobile() || widgetopts_is_mobile() || widgetopts_is_tablet()) {
                //mobile
                // if (!widgetopts_is_tablet()) {
                //     if (empty($opts['devices']['options']) || $opts['devices']['options'] == 'hide') {
                //         if (isset($opts['devices']['mobile']) && $opts['devices']['mobile'] == '1') {
                //             $hidden = true;
                //         }
                //     } else if ($opts['devices']['options'] == 'show') {
                //         if (!isset($opts['devices']['mobile']) || empty($opts['devices']['mobile'])) {
                //             $hidden = true;
                //         }
                //     }

                //     $hidden = apply_filters('widget_options_devices_mobile', $hidden);
                //     if ($hidden) {
                //         return false;
                //     }
                // } else {
                //     //tablet
                //     if (empty($opts['devices']['options']) || $opts['devices']['options'] == 'hide') {
                //         if (isset($opts['devices']['tablet']) && $opts['devices']['tablet'] == '1') {
                //             $hidden = true;
                //         }
                //     } else if ($opts['devices']['options'] == 'show') {
                //         if (!isset($opts['devices']['tablet']) || empty($opts['devices']['tablet'])) {
                //             $hidden = true;
                //         }
                //     }

                //     $hidden = apply_filters('widget_options_devices_tablet', $hidden);
                //     if ($hidden) {
                //         return false;
                //     }
                // }

                //for a time being filter
                if (empty($opts['devices']['options']) || $opts['devices']['options'] == 'hide') {
                    //if both tablet and mobile are set then hide the widget
                    if (
                        isset($opts['devices']['mobile']) && $opts['devices']['mobile'] == '1' &&
                        isset($opts['devices']['tablet']) && $opts['devices']['tablet'] == '1'
                    ) {
                        $hidden = true;
                    } else {
                        //else do nothing and css will make the final decision
                    }
                } else if ($opts['devices']['options'] == 'show') {
                    //if both tablet and mobile are set then hide the widget
                    if ((!isset($opts['devices']['mobile']) || empty($opts['devices']['mobile'])) &&
                        (!isset($opts['devices']['tablet']) || empty($opts['devices']['tablet']))
                    ) {
                        $hidden = true;
                    } else {
                        //else do nothing and css will make the final decision
                    }
                }

                $hidden = apply_filters('widget_options_devices_mobile', $hidden);
                if ($hidden) {
                    return false;
                }
            } else {
                //for desktop
                if (empty($opts['devices']['options']) || $opts['devices']['options'] == 'hide') {
                    if (isset($opts['devices']['desktop']) && $opts['devices']['desktop'] == '1') {
                        $hidden = true;
                    }
                } else if ($opts['devices']['options'] == 'show') {
                    if (!isset($opts['devices']['desktop']) || empty($opts['devices']['desktop'])) {
                        $hidden = true;
                    }
                }

                $hidden = apply_filters('widget_options_devices_desktop', $hidden);
                if ($hidden) {
                    return false;
                }
            }
        }

        $isWooPage = false;
        if (class_exists('WooCommerce')) {
            $wooPageID = 0;

            $wooPageID = (is_shop()) ? get_option('woocommerce_shop_page_id') : $wooPageID;
            if ($wooPageID) {
                $isWooPage = true;

                $visibility['pages'] = !empty($visibility['pages']) ? $visibility['pages'] : [];
                if ($visibility_opts == 'hide' && (array_key_exists($wooPageID, $visibility['pages']) || in_array($wooPageID, $visibility['pages']))) {
                    $hidden = true; //hide if exists on hidden pages
                } elseif ($visibility_opts == 'show' &&  !array_key_exists($wooPageID, $visibility['pages']) && !in_array($wooPageID, $visibility['pages'])) {
                    $hidden = true; //hide if doesn't exists on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_page', $hidden);

                if ($hidden) {
                    return false;
                }
            }
        }

        // Normal Pages
        if (!$isWooPage) {
            if ($is_misc && ((is_home() && is_front_page()) || is_front_page())) {
                if (isset($visibility['misc']['home']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide if checked on hidden pages
                } elseif (!isset($visibility['misc']['home']) && $visibility_opts == 'show') {
                    $hidden = true; //hide if not checked on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_home', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_misc && ((!is_front_page() && is_home()))) { //filter for blog page
                if (isset($visibility['misc']['blog']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide if checked on hidden pages
                } elseif (!isset($visibility['misc']['blog']) && $visibility_opts == 'show') {
                    $hidden = true; //hide if not checked on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_blog', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_tax && is_category()) {
                if (!isset($visibility['categories'])) {
                    $visibility['categories'] = array();
                }

                if (isset($visibility['tax_terms']['category'])) {
                    $visibility['categories'] = $visibility['tax_terms']['category'];
                }

                $selected_taxterms_page = 1;
                if (isset($visibility['tax_terms_page']) && isset($visibility['tax_terms_page']['category'])) {
                    $selected_taxterms_page = $visibility['tax_terms_page']['category'];
                }

                //for taxonomy category checking
                if (!isset($visibility['taxonomies'])) {
                    $visibility['taxonomies'] = array();
                }
                //check first the category taxonomy
                if ($visibility_opts == 'hide' && array_key_exists('category', $visibility['taxonomies'])) {
                    $hidden = true; //hide if exists on hidden pages
                } elseif ($visibility_opts == 'show' && !array_key_exists('category', $visibility['taxonomies'])) {
                    $hidden = true; //hide if doesn't exists on visible pages
                }

                // WPML TRANSLATION OBJECT FIX
                $category_id = ($hasWPML) ? apply_filters('wpml_object_id', get_query_var('cat'), 'category', true, $default_language) : get_query_var('cat');

                if ($visibility_opts == 'hide' && ((array_key_exists($category_id, $visibility['categories']) && $visibility['categories'][$category_id] == '1') || (in_array($category_id, $visibility['categories']) && ($selected_taxterms_page == 1 || $selected_taxterms_page == 2)) || ($is_misc && isset($visibility['misc']['archives'])))) {
                    $hidden = true; //hide if exists on hidden pages
                } elseif ($visibility_opts == 'show' && ((!array_key_exists($category_id, $visibility['categories']) && empty($visibility['categories'][$category_id])) && !(in_array($category_id, $visibility['categories']) && ($selected_taxterms_page == 1 || $selected_taxterms_page == 2)) && !($is_misc && isset($visibility['misc']['archives'])))) {
                    $hidden = true; //hide if doesn't exists on visible pages
                } elseif (((array_key_exists($category_id, $visibility['categories']) && $visibility['categories'][$category_id] == '1') || (in_array($category_id, $visibility['categories']) && ($selected_taxterms_page == 1 || $selected_taxterms_page == 2))) && $visibility_opts == 'hide') {
                    $hidden = true; //hide to all categories
                } elseif (((array_key_exists($category_id, $visibility['categories']) && $visibility['categories'][$category_id] == '1') || (in_array($category_id, $visibility['categories']) && ($selected_taxterms_page == 1 || $selected_taxterms_page == 2)) || ($is_misc && isset($visibility['misc']['archives']))) && $visibility_opts == 'show') {
                    $hidden = false; //hide to all categories
                }

                if ($visibility_opts == 'show' && array_key_exists('category', $visibility['taxonomies'])) {
                    $hidden = false; //hide if doesn't exists on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_categories', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_tax && is_tag()) {
                if (!isset($visibility['tags'])) {
                    $visibility['tags'] = array();
                }

                if (((isset($visibility['taxonomies']['post_tag']) || ($is_misc && isset($visibility['misc']['archives']))) && $visibility_opts == 'hide') ||
                    ((!isset($visibility['taxonomies']['post_tag']) && !($is_misc && isset($visibility['misc']['archives']))) && $visibility_opts == 'show')
                ) {
                    $hidden = true; //hide to all tags
                } elseif (isset($visibility['taxonomies']['post_tag']) && $visibility_opts == 'show') {
                    $hidden = false; //hide to all tags
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_tags', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_tax && is_tax()) {
                $term = get_queried_object();
                if (!isset($visibility['taxonomies'])) {
                    $visibility['taxonomies'] = array();
                }

                if ($visibility_opts == 'hide' && (!is_null($term) && isset($term->taxonomy) && (array_key_exists($term->taxonomy, $visibility['taxonomies'])) || ($is_misc && is_archive() && isset($visibility['misc']['archives'])))) {
                    $hidden = true; //hide if exists on hidden pages
                } elseif ($visibility_opts == 'show' && !(!is_null($term) && isset($term->taxonomy) && array_key_exists($term->taxonomy, $visibility['taxonomies'])) && !($is_misc && is_archive() && isset($visibility['misc']['archives']))) {
                    $hidden = true; //hide if doesn't exists on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_taxonomies', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_misc && is_archive()) {
                if (isset($visibility['misc']['archives']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide if checked on hidden pages
                } elseif (!isset($visibility['misc']['archives']) && $visibility_opts == 'show') {
                    $hidden = true; //hide if not checked on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_archives', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_misc && is_404()) {
                if (isset($visibility['misc']['404']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide if checked on hidden pages
                } elseif (!isset($visibility['misc']['404']) && $visibility_opts == 'show') {
                    $hidden = true; //hide if not checked on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_404', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif ($is_misc && is_search()) {
                if (isset($visibility['misc']['search']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide if checked on hidden pages
                } elseif (!isset($visibility['misc']['search']) && $visibility_opts == 'show') {
                    $hidden = true; //hide if not checked on visible pages
                }

                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_search', $hidden);
                if ($hidden) {
                    return false;
                }
            } elseif (is_single() && !is_page()) {
                global $post;
                $type = '';
                if (!$post) {
                    $current_post = get_post();
                    $type = $current_post->post_type;
                } else {
                    $type = $post->post_type;
                }

                if ($is_misc) {
                    if (isset($visibility['misc']['single']) && $visibility_opts == 'show') {
                        return $instance;
                    }
                }

                if (isset($visibility['tax_terms']['category'])) {
                    $visibility['categories'] = $visibility['tax_terms']['category'];
                }

                $selected_taxterms_page = 1;
                if (isset($visibility['tax_terms_page']) && isset($visibility['tax_terms_page']['category'])) {
                    $selected_taxterms_page = $visibility['tax_terms_page']['category'];
                }

                if (!isset($visibility['types'])) {
                    $visibility['types'] = array();
                }
                if ($visibility_opts == 'hide' && (array_key_exists($type, $visibility['types']) || ($is_misc && isset($visibility['misc']['single'])))) {
                    $hidden = true; //hide if exists on hidden pages
                } elseif ($visibility_opts == 'show' && (!array_key_exists($type, $visibility['types']) && (($is_misc && (!isset($visibility['misc']['single']))) || !$is_misc))) {
                    $hidden = true; //hide if doesn't exists on visible pages
                }

                // do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_types', $hidden);
                //hide posts assign on category
                if (!isset($visibility['categories'])) {
                    $visibility['categories'] = array();
                }
                if (isset($visibility['categories']['all_categories']) && $visibility_opts == 'hide') {
                    $hidden = true; //hide to all categories
                } elseif (isset($visibility['categories']['all_categories']) && $visibility_opts == 'show') {
                    $hidden = false; //hide to all categories
                } elseif (!isset($visibility['categories']['all_categories']) && !empty($visibility['categories'])) {
                    $cats           = wp_get_post_categories(get_the_ID());
                    if (is_array($cats) && !empty($cats)) {
                        $checked_cats   = array_keys($visibility['categories']);
                        $checked_cats = (intval($checked_cats[0]) == 0) ? $visibility['categories'] : $checked_cats;
                        $intersect      = array_intersect($cats, $checked_cats);
                        if (!empty($intersect) && $visibility_opts == 'hide') {
                            if ($selected_taxterms_page == 1 || $selected_taxterms_page == 3) {
                                $hidden = true;
                            }
                        } elseif (!empty($intersect) && $visibility_opts == 'show') {
                            if ($selected_taxterms_page == 1 || $selected_taxterms_page == 3) {
                                $hidden = false;
                            }
                        }
                    }
                }
                // do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_post_category', $hidden);
                if ($hidden) {
                    return false;
                }
                // echo $type;
            } elseif ($is_types && (is_page() || get_post_type(get_the_ID()) == 'page')) {
                global $post;

                // WPML FIX
                $page_id = get_queried_object_id();
                $parent_id = wp_get_post_parent_id($page_id);

                $pageID = ($hasWPML) ? apply_filters('wpml_object_id', $page_id, 'page', true, $default_language) : $page_id;
                $parentID = ($hasWPML) ? apply_filters('wpml_object_id', $parent_id, 'page', true, $default_language) : $parent_id;

                //do post type condition first
                if (isset($visibility['types']) && isset($visibility['types']['page'])) {
                    if ($visibility_opts == 'hide' && array_key_exists('page', $visibility['types'])) {
                        $hidden = true; //hide if exists on hidden pages
                    } elseif ($visibility_opts == 'show' && !array_key_exists('page', $visibility['types'])) {
                        $hidden = true; //hide if doesn't exists on visible pages
                    }
                } else {
                    //do per pages condition
                    if (!isset($visibility['pages'])) {
                        $visibility['pages'] = array();
                    }

                    $page_in_array = in_array($pageID, $visibility['pages']);
                    //for the compatibility of the data of lower version 3.8.10 and below
                    if (array_key_exists($pageID, $visibility['pages'])) {
                        if ($visibility['pages'][$pageID] == 1) {
                            $page_in_array = true;
                        }
                    }

                    if ($visibility_opts == 'hide' && $page_in_array) {
                        $hidden = true; //hide if exists on hidden pages
                    } elseif ($visibility_opts == 'show' && $pageID == 0) {
                        //for old versin v3.8.10 and below, need to resave the widget option
                        if (!in_array($pageID, $visibility['pages'])) {
                            $hidden = true;
                        }
                    } elseif ($visibility_opts == 'show' && !$page_in_array) {
                        $hidden = true; //hide if doesn't exists on visible pages
                    }
                }
                //do return to bypass other conditions
                $hidden = apply_filters('widget_options_visibility_page', $hidden);
                if ($hidden) {
                    return false;
                }
            }
        }
        //end wordpress pages


        //ACF
        if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
            if (isset($visibility['acf']['field']) && !empty($visibility['acf']['field'])) {
                $acf = get_field_object($visibility['acf']['field']);

                if ($acf && is_array($acf)) {
                    $acf_visibility    = (isset($visibility['acf']) && isset($visibility['acf']['visibility'])) ? $visibility['acf']['visibility'] : 'hide';

                    //handle repeater fields
                    if (isset($acf['value'])) {
                        if (is_array($acf['value'])) {
                            $acf['value'] = implode(', ', array_map(function ($acf_array_value) {
                                $acf_implode = '';
                                if (is_array($acf_array_value)) {
                                    $acf_implode = implode(',', array_filter($acf_array_value));
                                } else {
                                    $acf_implode = $acf_array_value;
                                }
                                return $acf_implode;
                            }, $acf['value']));
                        }
                    }

                    switch ($visibility['acf']['condition']) {
                        case 'equal':
                            if (isset($acf['value'])) {
                                if ('show' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
                                    $hidden = false;
                                } else if ('show' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
                                    $hidden = true;
                                } else if ('hide' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
                                    $hidden = true;
                                } else if ('hide' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
                                    $hidden = false;
                                }
                            }
                            break;

                        case 'not_equal':
                            if (isset($acf['value'])) {
                                if ('show' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
                                    $hidden = true;
                                } else if ('show' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
                                    $hidden = false;
                                } else if ('hide' == $acf_visibility && $acf['value'] == $visibility['acf']['value']) {
                                    $hidden = false;
                                } else if ('hide' == $acf_visibility && $acf['value'] != $visibility['acf']['value']) {
                                    $hidden = true;
                                }
                            }
                            break;

                        case 'contains':
                            if (isset($acf['value'])) {
                                if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
                                    $hidden = false;
                                } else if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
                                    $hidden = true;
                                } else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
                                    $hidden = true;
                                } else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
                                    $hidden = false;
                                }
                            }
                            break;

                        case 'not_contains':
                            if (isset($acf['value'])) {
                                if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
                                    $hidden = true;
                                } else if ('show' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
                                    $hidden = false;
                                } else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) !== false) {
                                    $hidden = false;
                                } else if ('hide' == $acf_visibility && strpos($acf['value'], $visibility['acf']['value']) === false) {
                                    $hidden = true;
                                }
                            }
                            break;

                        case 'empty':
                            if ('show' == $acf_visibility && empty($acf['value'])) {
                                $hidden = false;
                            } else if ('show' == $acf_visibility && !empty($acf['value'])) {
                                $hidden = true;
                            } elseif ('hide' == $acf_visibility && empty($acf['value'])) {
                                $hidden = true;
                            } else if ('hide' == $acf_visibility && !empty($acf['value'])) {
                                $hidden = false;
                            }
                            break;

                        case 'not_empty':
                            if ('show' == $acf_visibility && empty($acf['value'])) {
                                $hidden = true;
                            } else if ('show' == $acf_visibility && !empty($acf['value'])) {
                                $hidden = false;
                            } elseif ('hide' == $acf_visibility && empty($acf['value'])) {
                                $hidden = false;
                            } else if ('hide' == $acf_visibility && !empty($acf['value'])) {
                                $hidden = true;
                            }
                            break;

                        default:
                            # code...
                            break;
                    }

                    // //do return to bypass other conditions
                    $hidden = apply_filters('widget_options_visibility_acf', $hidden);
                    if ($hidden) {
                        return false;
                    }
                }
            }
        }

        //login state
        if (isset($widget_options['state']) && 'activate' == $widget_options['state'] && isset($opts['roles'])) {
            if (isset($opts['roles']['state']) && !empty($opts['roles']['state'])) {
                //do state action here
                if ($opts['roles']['state'] == 'out' && is_user_logged_in()) {
                    return false;
                } else if ($opts['roles']['state'] == 'in' && !is_user_logged_in()) {
                    return false;
                }
            }
        }

        if ('activate' == $widget_options['logic']) {
            // display widget logic
            if (isset($opts['class']) && isset($opts['class']['logic']) && !empty($opts['class']['logic'])) {
                $display_logic = stripslashes(trim($opts['class']['logic']));
                $display_logic = apply_filters('widget_options_logic_override', $display_logic);
                $display_logic = apply_filters('extended_widget_options_logic_override', $display_logic);
                if ($display_logic === false) {
                    return false;
                }
                if ($display_logic === true) {
                    return true;
                }
                // if (stristr($display_logic, "return") === false) {
                //     $display_logic = "return (" . $display_logic . ");";
                // }
                $display_logic = htmlspecialchars_decode($display_logic, ENT_QUOTES);
                if (!widgetopts_safe_eval($display_logic)) {
                    return false;
                }
            }
        }

        if ('activate' == $widget_options['hide_title']) {
            //hide widget title
            if (isset($instance['title']) && isset($opts['class']) && isset($opts['class']['title']) && '1' == $opts['class']['title']) {
                $instance['title'] = '';
            }
        }

        return $instance;
    }
    add_filter('widget_display_callback', 'widgetopts_display_callback', 50, 3);
endif;

//Don't show widget title
if (!function_exists('widgetopts_remove_title')) :
    function widgetopts_remove_title($widget_title, $instance = array(), $widget_id = '')
    {
        global $widget_options;
        if ('activate' == $widget_options['hide_title'] && is_array($instance) && !empty($instance)) {
            foreach ($instance as $key => $value) {
                if (substr($key, 0, 20) == 'extended_widget_opts') {
                    $opts       = (isset($instance[$key])) ? (array)$instance[$key] : array();

                    if (isset($opts['class']) && isset($opts['class']['title']) && '1' == $opts['class']['title']) {
                        return;
                    }

                    break;
                }
            }
            return $widget_title;
        } else {
            return ($widget_title);
        }
    }
    add_filter('widget_title', 'widgetopts_remove_title', 10, 4);
endif;

/*
 * Add custom classes on dynamic_sidebar_params filter
 */
if (!function_exists('widgetopts_add_classes')) :
    function widgetopts_add_classes($params)
    {
        global $widget_options, $wp_registered_widget_controls;
        $classe_to_add  = '';
        $id_base        = $wp_registered_widget_controls[$params[0]['widget_id']]['id_base'];
        $instance       = get_option('widget_' . $id_base);

        $num = substr($params[0]['widget_id'], -1);
        if (isset($wp_registered_widget_controls[$params[0]['widget_id']]['params'][0]['number'])) {
            $num = $wp_registered_widget_controls[$params[0]['widget_id']]['params'][0]['number'];
        } elseif (isset($wp_registered_widget_controls[$params[0]['widget_id']]['callback']) && is_array($wp_registered_widget_controls[$params[0]['widget_id']]['callback'])) {
            if (isset($wp_registered_widget_controls[$params[0]['widget_id']]['callback'][0]) && isset($wp_registered_widget_controls[$params[0]['widget_id']]['callback'][0]->number)) {
                $num = $wp_registered_widget_controls[$params[0]['widget_id']]['callback'][0]->number;
            }
        }
        if (isset($instance[$num])) {
            $opts           = (isset($instance[$num]['extended_widget_opts-' . $params[0]['widget_id']])) ? $instance[$num]['extended_widget_opts-' . $params[0]['widget_id']] : array();
            if (empty($opts) && isset($instance[$num]['content']) && !empty($instance[$num]['content'])) {
                /* if $opts is empty, try to get data from blocks */
                $block = parse_blocks($instance[$num]['content']);
                if (!empty($block[0]) && !empty($block[0]['attrs'])) {
                    if (!empty($block[0]['attrs']['extended_widget_opts'])) {
                        $opts = $block[0]['attrs']['extended_widget_opts'];
                    }
                }
            }
        } else {
            $opts = array();
        }

        $custom_class   = isset($opts['class']) ? $opts['class'] : '';
        $widget_id_set  = $params[0]['widget_id'];

        if ('activate' == $widget_options['classes'] && isset($widget_options['settings']['classes'])) {
            //don't add the IDs when the setting is set to NO
            if (isset($widget_options['settings']['classes']['id'])) {
                if (is_array($custom_class) && isset($custom_class['id']) && !empty($custom_class['id'])) {
                    $params[0]['before_widget'] = preg_replace('/id="[^"]*/', "id=\"{$custom_class['id']}", $params[0]['before_widget'], 1);
                    $widget_id_set = $custom_class['id'];
                }
            }
        }

        $get_classes = widgetopts_classes_generator($opts, $widget_options, $widget_options['settings']);

        //double check array
        if (!is_array($get_classes)) {
            $get_classes = array();
        }

        if ('activate' == $widget_options['classes']) {
            if (isset($widget_options['settings']['classes']['auto'])) {
                //do nothing
            } else {
                //check if widget class exists
                if ((strpos($params[0]['before_widget'], '"widget ') !== false) ||
                    (strpos($params[0]['before_widget'], ' widget ') !== false) ||
                    (strpos($params[0]['before_widget'], ' widget"') !== false)
                ) {
                    //do nothing
                } else {
                    $get_classes[] = 'widget';
                }
            }
        }

        if (!empty($get_classes)) {
            $classes        = 'class="' . (implode(' ', $get_classes)) . ' ';
            $params[0]['before_widget'] = str_replace('class="', $classes, $params[0]['before_widget']);
        }

        // $params[0]['before_widget'] = str_replace('class="', ' data-animation="asdf" class="', $params[0]['before_widget']);

        return $params;
    }
    add_filter('dynamic_sidebar_params', 'widgetopts_add_classes');
endif;
