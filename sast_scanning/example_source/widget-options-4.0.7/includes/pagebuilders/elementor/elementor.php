<?php

/**
 * Extends funtionality to Elementor Pagebuilder
 *
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.3
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;

if (!function_exists('widgetopts_elementor_section')) {
    //Add "Widget Options" section to every Elementor Widgets
    add_action('elementor/element/after_section_end', 'widgetopts_elementor_section', 5, 3);
    function widgetopts_elementor_section($element, $section_id, $args)
    {
        if (Elementor\Plugin::$instance->editor->is_edit_mode()) {
            global $widget_options;

            //filter the elements first to avoid conflicts that can cause pagebuilder not to load
            if (!in_array($element->get_name(), array('global-settings', 'section', 'page-settings', 'oew-blog-grid'))) {

                //create array of section_id to set Widget Options Section to single section to avoid issues
                $widgetopts_elementor_section_id = apply_filters(
                    'widgetopts_elementor_section_id',
                    array(
                        'section_image',
                        'section_advanced',
                        'section_title',
                        'section_editor',
                        'section_video',
                        'section_button',
                        'section_divider',
                        'section_spacer',
                        'section_map',
                        'section_icon',
                        'section_gallery',
                        'section_image_carousel',
                        'section_icon_list',
                        'section_counter',
                        'section_testimonial',
                        'section_tabs',
                        'section_toggle',
                        'section_social_icon',
                        'section_alert',
                        'section_audio',
                        'section_shortcode',
                        'section_anchor',
                        'section_sidebar',
                        'section_layout',
                        'section_slides',
                        'section_form_fields',
                        'section_list',
                        'section_header',
                        'section_pricing',
                        'section_countdown',
                        'section_buttons_content',
                        'section_blockquote_content',
                        'section_content',
                        'section_login_content',
                        'text_elements',
                        'section_side_a_content',
                        'section_side_b_content',
                        '_section_style'
                    )
                );

                //filter by the section_ids above
                if (in_array($section_id, $widgetopts_elementor_section_id)) {
                    $element->start_controls_section(
                        'widgetopts_section',
                        [
                            'tab'   => Elementor\Controls_Manager::TAB_ADVANCED,
                            'label' => __('Widget Options', 'widget-options'),
                        ],
                        [
                            'overwrite'         => true
                        ]
                    );

                    $element->start_controls_tabs('widgetopts_content_tabs', [
                        'overwrite'         => true
                    ]);

                    if (isset($widget_options['visibility']) && 'activate' == $widget_options['visibility']) {
                        widgetopts_elementor_tab_visibility($element, $section_id, $args);
                    }

                    if (isset($widget_options['state']) && 'activate' == $widget_options['state']) {
                        widgetopts_elementor_tab_state($element, $section_id, $args);
                    }

                    if (current_user_can('administrator') && ('activate' == $widget_options['logic'] || (isset($widget_options['sliding']) && 'activate' == $widget_options['sliding'] && in_array($element->get_name(), array('button', 'button_plus', 'eael-creative-button', 'cta'))))) {
                        widgetopts_elementor_tab_settings($element, $section_id, $args);
                    }

                    //upsell pro
                    if (!is_plugin_active('extended-widget-options/plugin.php')) {
                        $element->start_controls_tab(
                            'widgetopts_tab_upsell',
                            [
                                'label' => __('<i class="dashicons dashicons-plus"></i>', 'widget-options')
                            ],
                            [
                                'overwrite'         => true
                            ]
                        );
                        $upgrade_link = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . '?utm_source=elementor&utm_medium=upgrade&utm_campaign=upgradebtn');
                        $element->add_control(
                            'widgetopts_pro',
                            [
                                'type' => Elementor\Controls_Manager::RAW_HTML,
                                'raw' => '<div class="elementor-panel-nerd-box">
                                    <i class="elementor-panel-nerd-box-icon dashicons dashicons-lock"></i>
                    						<div class="elementor-panel-nerd-box-title">' .
                                    __('Unlock All Widget Options', 'widget-options') .
                                    '</div>
                    						<div class="elementor-panel-nerd-box-message">' .
                                    __('Upgrade to Widget Options Extended to unlock all options to easily control and manage each Elementor widget.', 'widget-options') .
                                    '</div>
                    						<a href="' . $upgrade_link . '" class="elementor-panel-nerd-box-link elementor-button elementor-button-default elementor-go-pro" target="_blank">' .
                                    __('Upgrade Now!', 'widget-options') .
                                    '</a>
                    						</div>',
                            ],
                            [
                                'overwrite'         => true
                            ]
                        );
                        $element->end_controls_tab();
                    }
                    $element->end_controls_tabs();

                    $element->end_controls_section();
                }
            }
        }
    }
}

if (!function_exists('widgetopts_elementor_tab_visibility')) {
    function widgetopts_elementor_tab_visibility($element, $section_id, $args)
    {
        global $widget_options, $widgetopts_taxonomies, $widgetopts_pages, $widgetopts_types, $widgetopts_categories;

        $pages      = (!empty($widgetopts_pages))       ? $widgetopts_pages         : array();
        $taxonomies = (!empty($widgetopts_taxonomies))  ? $widgetopts_taxonomies    : array();
        $types      = (!empty($widgetopts_types))       ? $widgetopts_types         : array();
        $categories = (!empty($widgetopts_categories))  ? $widgetopts_categories    : array();

        // print_r( $get_terms['community-category'] );

        $element->start_controls_tab(
            'widgetopts_tab_visibility',
            [
                'label' => __('<span class="dashicons dashicons-visibility"></span>', 'widget-options')
            ],
            [
                'overwrite'         => true
            ]
        );

        $element->add_control(
            'widgetopts_visibility',
            [
                'label'         => __('Show/Hide', 'widget-options'),
                'type'          => Elementor\Controls_Manager::SELECT,
                'default'       => 'hide',
                'options'       => [
                    'show' => __('Show on Selected Pages'),
                    'hide' => __('Hide on Selected Pages')
                ],
                // 'separator'     => 'none'
            ],
            [
                'overwrite'         => true
            ]
        );

        if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['post_type']) && '1' == $widget_options['settings']['visibility']['post_type']) {
            $pages_array = array();
            if (!empty($pages)) {
                foreach ($pages as $page) {
                    $pages_array[$page->ID] = $page->post_title;
                }

                $element->add_control(
                    'widgetopts_pages',
                    [
                        'label'             => __('Pages', 'widget-options'),
                        'type'              => Elementor\Controls_Manager::SELECT2,
                        'multiple'          => true,
                        'label_block'       => true,
                        'separator'         => 'before',
                        'options'           => $pages_array,
                        'render_type'       => 'none',
                        'description'       => __('Click on the field to search and select pages', 'widget-options')
                    ],
                    [
                        'overwrite'         => true
                    ]
                );
            }

            if (!empty($types)) {
                $types_array = array();
                foreach ($types as $ptype => $type) {
                    $types_array[$ptype] = $type->labels->name;
                }

                $element->add_control(
                    'widgetopts_types',
                    [
                        'label'             => __('Post Types', 'widget-options'),
                        'type'              => Elementor\Controls_Manager::SELECT2,
                        'multiple'          => true,
                        'label_block'       => true,
                        'separator'         => 'before',
                        'options'           => $types_array,
                        'render_type'       => 'none',
                        'description'       => __('Click on the field to search and select custom post types', 'widget-options')
                    ],
                    [
                        'overwrite'         => true
                    ]
                );
            }
        }

        if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['taxonomies']) && '1' == $widget_options['settings']['visibility']['taxonomies']) {

            if (!empty($categories)) {
                $cat_array = array();
                foreach ($categories as $cat) {
                    $cat_array[$cat->cat_ID] = $cat->cat_name;
                }

                $element->add_control(
                    'widgetopts_tax_category',
                    [
                        'label'             => __('Categories', 'widget-options'),
                        'type'              => Elementor\Controls_Manager::SELECT2,
                        'multiple'          => true,
                        'label_block'       => true,
                        'separator'         => 'before',
                        'options'           => $cat_array,
                        'render_type'       => 'none',
                        'description'       => __('Click on the field to search and select categories', 'widget-options')
                    ],
                    [
                        'overwrite'         => true
                    ]
                );
            }

            if (!empty($taxonomies)) {
                $tax_array = array();
                foreach ($taxonomies as $taxonomy) {
                    $tax_array[$taxonomy->name] = $taxonomy->label;
                }

                $element->add_control(
                    'widgetopts_taxonomies',
                    [
                        'label'             => __('Taxonomies', 'widget-options'),
                        'type'              => Elementor\Controls_Manager::SELECT2,
                        'multiple'          => true,
                        'label_block'       => true,
                        'separator'         => 'before',
                        'options'           => $tax_array,
                        'render_type'       => 'none',
                        'description'       => __('Click on the field to search and select taxonomies', 'widget-options')
                    ],
                    [
                        'overwrite'         => true
                    ]
                );
            }
        }

        if (isset($widget_options['settings']['visibility']) && isset($widget_options['settings']['visibility']['misc']) && '1' == $widget_options['settings']['visibility']['misc']) {
            $element->add_control(
                'widgetopts_misc',
                [
                    'label'             => __('Miscellaneous', 'widget-options'),
                    'type'              => Elementor\Controls_Manager::SELECT2,
                    'multiple'          => true,
                    'label_block'       => true,
                    'separator'         => 'before',
                    'options'           => [
                        'home'      =>  __('Home/Front', 'widget-options'),
                        'blog'      =>  __('Blog', 'widget-options'),
                        'archives'  =>  __('Archives', 'widget-options'),
                        '404'       =>  __('404', 'widget-options'),
                        'search'    =>  __('Search', 'widget-options')
                    ],
                    'render_type'       => 'none',
                    'description'       => __('Click on the field to search and select miscellaneous pages', 'widget-options')
                ],
                [
                    'overwrite'         => true
                ]
            );
        }
        $element->end_controls_tab();
    }
}

if (!function_exists('widgetopts_elementor_tab_state')) {
    function widgetopts_elementor_tab_state($element, $section_id, $args)
    {
        global $widget_options;

        $element->start_controls_tab(
            'widgetopts_tab_state',
            [
                'label' => __('<span class="dashicons dashicons-admin-users"></span>', 'widget-options')
            ],
            [
                'overwrite'         => true
            ]
        );

        $element->add_control(
            'widgetopts_roles_state',
            [
                'label'         => __('User Login State', 'widget-options'),
                'type'          => Elementor\Controls_Manager::SELECT,
                'default'       => 'hide',
                'options'       => [
                    ''     => __('Select Visibility Option'),
                    'in' => __('Show only for Logged-in Users'),
                    'out' => __('Show only for Logged-out Users')
                ],
                'description'   => __('Restrict widget visibility for logged-in and logged-out users.', 'widget-options')
            ],
            [
                'overwrite'         => true
            ]
        );

        $element->end_controls_tab();
    }
}

if (!function_exists('widgetopts_elementor_tab_settings')) {
    function widgetopts_elementor_tab_settings($element, $section_id, $args)
    {
        global $widget_options;

        $element->start_controls_tab(
            'widgetopts_tab_settings',
            [
                'label' => __('<span class="dashicons dashicons-admin-generic"></span>', 'widget-options')
            ],
            [
                'overwrite'         => true
            ]
        );

        if (is_plugin_active('sliding-widget-options/plugin.php') && 'activate' == $widget_options['sliding'] && in_array($element->get_name(), array('button', 'button_plus', 'eael-creative-button', 'cta'))) {
            $element->add_control(
                'widgetopts_open_sliding',
                [
                    'label'          => __('Open Pop-up or Sliding Widgets', 'widget-options'),
                    'type'           => Elementor\Controls_Manager::SWITCHER,
                    'return_value'   => 'on',
                    'default'        => '',
                    'separator'      => 'none'
                ],
                [
                    'overwrite'         => true
                ]
            );
        }

        if ('activate' == $widget_options['logic'] && current_user_can('administrator')) {

            $element->add_control(
                'widgetopts_logic',
                [
                    'type'          => Elementor\Controls_Manager::TEXTAREA,
                    'label'         => __('Display Widget Logic', 'widget-options'),
                    'description'   => __('Add your PHP Conditional Tags. Please note that this will be EVAL\'d directly.', 'widget-options'),
                    // 'separator'     => 'none',
                ],
                [
                    'overwrite'         => true
                ]
            );
        }


        if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) {
            $fields = array();

            if (function_exists('acf_get_field_groups')) {
                $groups = acf_get_field_groups();
                if (is_array($groups)) {
                    foreach ($groups as $group) {
                        $fields_group = acf_get_fields($group);
                        if (!empty($fields_group)) {
                            foreach ($fields_group as $k => $fg) {
                                $fields[$fg['key']] = $fg['label'];
                            }
                        }
                    }
                }
            } else {
                $groups = apply_filters('acf/get_field_groups', array());
                if (is_array($groups)) {
                    foreach ($groups as $group) {
                        $fields_group = apply_filters('acf/field_group/get_fields', array(), $group['id']);
                        if (!empty($fields_group)) {
                            foreach ($fields_group as $k => $fg) {
                                $fields[$fg['key']] = $fg['label'];
                            }
                        }
                    }
                }
            }

            $element->add_control(
                'widgetopts_acf_title',
                [
                    'type' => Elementor\Controls_Manager::RAW_HTML,
                    'separator'         => 'before',
                    'raw' => '<h3>' . __('Advanced Custom Fields', 'widget-options') . '</h3>',
                ],
                [
                    'overwrite'         => true
                ]
            );

            $element->add_control(
                'widgetopts_acf_visibility',
                [
                    'label'         => __('Show/Hide', 'widget-options'),
                    'type'          => Elementor\Controls_Manager::SELECT,
                    'default'       => 'hide',
                    'options'       => [
                        'show' => __('Show when Condition\'s Met'),
                        'hide' => __('Hide when Condition\'s Met')
                    ],
                    'separator'         => 'before',
                ],
                [
                    'overwrite'         => true
                ]
            );

            $element->add_control(
                'widgetopts_acf_field',
                [
                    'label'             => __('Select ACF Field', 'widget-options'),
                    'type'              => Elementor\Controls_Manager::SELECT2,
                    'multiple'          => false,
                    'label_block'       => true,
                    'options'           => $fields,
                    'render_type'       => 'none',
                    'description'       => __('Select ACF field.', 'widget-options')
                ],
                [
                    'overwrite'         => true
                ]
            );

            $element->add_control(
                'widgetopts_acf_condition',
                [
                    'label'             => __('Condition', 'widget-options'),
                    'type'              => Elementor\Controls_Manager::SELECT2,
                    'multiple'          => false,
                    'label_block'       => true,
                    'options'           => [
                        'equal'      =>  __('Is Equal To', 'widget-options'),
                        'not_equal'  =>  __('Is Not Equal To', 'widget-options'),
                        'contains'   =>  __('Contains', 'widget-options'),
                        'not_contains'   =>  __('Does Not Contain', 'widget-options'),
                        'empty'      =>  __('Is Empty', 'widget-options'),
                        'not_empty'  =>  __('Is Not Empty', 'widget-options')
                    ],
                    'render_type'       => 'none',
                    'description'       => __('Select your condition for this widget visibility.', 'widget-options')
                ],
                [
                    'overwrite'         => true
                ]
            );
            $element->add_control(
                'widgetopts_acf',
                [
                    'type'          => Elementor\Controls_Manager::TEXTAREA,
                    'label'         => __('Conditional Value', 'widget-options'),
                    'description'   => __('Add your Conditional Value here if you selected Equal to, Not Equal To or Contains on the selection above.', 'widget-options'),
                    // 'separator'     => 'none',
                ],
                [
                    'overwrite'         => true
                ]
            );
        }

        $element->end_controls_tab();
    }
}
