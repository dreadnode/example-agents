<?php

/**
 * Pages Visibility Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Visibility Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for visibility options
 */
function widgetopts_tab_visibility($args)
{ ?>
    <li class="extended-widget-opts-tab-visibility">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-visibility" title="<?php _e('Page Visibility', 'widget-options'); ?>"><span class="dashicons dashicons-visibility"></span> <span class="tabtitle"><?php _e('Page Visibility', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_visibility', 1);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for visibility options
 */
function widgetopts_tabcontent_visibility($args)
{
    global $widget_options, $widgetopts_taxonomies, $widgetopts_pages, $widgetopts_types, $widgetopts_categories;

    $checked    = "";
    $main       = "";
    $selected   = 0;
    $tax_opts   = (array) get_option('extwopts_taxonomy_settings');
    $pages      = (!empty($widgetopts_pages))       ? $widgetopts_pages         : array();
    $taxonomies = (!empty($widgetopts_taxonomies))  ? $widgetopts_taxonomies    : array();
    $types      = (!empty($widgetopts_types))       ? $widgetopts_types         : array();
    $categories = (!empty($widgetopts_categories))  ? $widgetopts_categories    : array();

    //declare miscellaneous pages - wordpress default pages
    $misc       = array(
        'home'      =>  __('Home/Front', 'widget-options'),
        'blog'      =>  __('Blog', 'widget-options'),
        'archives'  =>  __('Archives', 'widget-options'),
        // 'single'    =>  __( 'Single Post', 'widget-options' ),
        '404'       =>  __('404', 'widget-options'),
        'search'    =>  __('Search', 'widget-options')
    );

    //unset builtin post types
    foreach (array('revision', 'attachment', 'nav_menu_item') as $unset) {
        unset($types[$unset]);
    }

    //pro version only
    // $get_terms = array();
    // if( !empty( $widget_options['settings']['taxonomies'] ) && is_array( $widget_options['settings']['taxonomies'] ) ){
    //     foreach ( $widget_options['settings']['taxonomies'] as $tax_opt => $vall ) {
    //         $tax_name = 'widgetopts_taxonomy_'. $tax_opt;
    //         global $$tax_name;
    //         $get_terms[ $tax_opt ] = $$tax_name;
    //     }
    // }


    //get save values
    $options_values = '';
    $misc_values    = array();
    $pages_values   = array();
    $types_values   = array();
    $cat_values     = array();
    $tax_values     = array();
    $terms_values   = array();
    $acf_values   = array();
    if (isset($args['params']) && isset($args['params']['visibility'])) {

        if (isset($args['params']['visibility']['options'])) {
            $options_values = $args['params']['visibility']['options'];
        }

        if (isset($args['params']['visibility']['misc'])) {
            $misc_values = $args['params']['visibility']['misc'];
        }

        if (isset($args['params']['visibility']['pages'])) {
            $pages_values = $args['params']['visibility']['pages'];
        }

        if (isset($args['params']['visibility']['types'])) {
            $types_values = $args['params']['visibility']['types'];
        }

        if (isset($args['params']['visibility']['categories'])) {
            $cat_values = $args['params']['visibility']['categories'];
        }

        if (isset($args['params']['visibility']['taxonomies'])) {
            $tax_values = $args['params']['visibility']['taxonomies'];
        }

        if (isset($args['params']['visibility']['tax_terms']) && isset($args['params']['visibility']['tax_terms']['category'])) {
            //$terms_values = $args['params']['visibility']['tax_terms']['category'];
            $cat_values = $args['params']['visibility']['tax_terms']['category'];
        }

        if (isset($args['params']['visibility']['acf'])) {
            $acf_values = $args['params']['visibility']['acf'];
        }

        if (isset($args['params']['visibility']['selected'])) {
            $selected = $args['params']['visibility']['selected'];
        }

        if (isset($args['params']['visibility']['main'])) {
            $main = $args['params']['visibility']['main'];
        }
    }

    // fix values for older settings
    $tmpPages_values = array();
    foreach ($pages_values as $objKey => $objPage) {
        if (isset($pages_values[$objKey]) && $pages_values[$objKey] == '1') {
            $tmpPages_values[] = $objKey;
        } else {
            $tmpPages_values[] = $objPage;
        }
    }
    $pages_values = $tmpPages_values;

    // fix values for older settings
    $tmpTerms_values = array();
    foreach ($cat_values as $objKey => $objTerm) {
        if (isset($cat_values[$objKey]) && $cat_values[$objKey] == '1') {
            $_objKey = $objKey;
            if (is_numeric($objKey)) {
                if (intval($objKey) == 0) {
                    $_objKey = 1;
                }
            }

            $tmpTerms_values[] = $_objKey;
        } else {
            $_objTerm = $objTerm;
            if (is_numeric($objTerm)) {
                if (intval($objTerm) == 0) {
                    $_objTerm = 1;
                }
            }

            $tmpTerms_values[] = $_objTerm;
        }
    }
    $term_values = $tmpTerms_values;
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-visibility" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-visibility">

        <div class="extended-widget-opts-styling-tabs extended-widget-opts-inside-tabs">
            <input type="hidden" id="extended-widget-opts-visibility-m-selectedtab" value="<?php echo $main; ?>" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][main]" />


            <div id="extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-main" class="extended-widget-opts-visibility-tabcontent extended-widget-opts-inside-tabcontent extended-widget-opts-inner-tabcontent">
                <p><strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
                    <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][options]">
                        <option value="hide" <?php if ($options_values == 'hide') {
                                                    echo 'selected="selected"';
                                                } ?>><?php _e('Hide on checked pages', 'widget-options'); ?></option>
                        <option value="show" <?php if ($options_values == 'show') {
                                                    echo 'selected="selected"';
                                                } ?>><?php _e('Show on checked pages', 'widget-options'); ?></option>
                    </select>
                </p>

                <div class="extended-widget-opts-visibility-tabs extended-widget-opts-inside-tabs">
                    <input type="hidden" id="extended-widget-opts-visibility-selectedtab" value="<?php echo $selected; ?>" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][selected]" />
                    <!--  start tab nav -->
                    <ul class="extended-widget-opts-visibility-tabnav-ul">
                        <?php if (
                            isset($widget_options['settings']['visibility']) &&
                            isset($widget_options['settings']['visibility']['misc']) &&
                            '1' == $widget_options['settings']['visibility']['misc']
                        ) { ?>
                            <li class="extended-widget-opts-visibility-tab-visibility">
                                <a href="#extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-misc" title="<?php _e('Pages', 'widget-options'); ?>"><span class="dashicons dashicons-admin-page"></span> <span class="tabtitle"><?php _e('Pages', 'widget-options'); ?></span></a>
                            </li>
                        <?php } ?>

                        <?php if (
                            isset($widget_options['settings']['visibility']) &&
                            isset($widget_options['settings']['visibility']['post_type']) &&
                            '1' == $widget_options['settings']['visibility']['post_type']
                        ) { ?>
                            <li class="extended-widget-opts-visibility-tab-visibility">
                                <a href="#extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-types" title="<?php _e('Post Types', 'widget-options'); ?>"><span class="dashicons dashicons-admin-post"></span> <span class="tabtitle"><?php _e('Post Types', 'widget-options'); ?></span></a>
                            </li>
                        <?php } ?>

                        <?php if (
                            isset($widget_options['settings']['visibility']) &&
                            isset($widget_options['settings']['visibility']['taxonomies']) &&
                            '1' == $widget_options['settings']['visibility']['taxonomies']
                        ) { ?>
                            <li class="extended-widget-opts-visibility-tab-visibility">
                                <a href="#extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-tax" title="<?php _e('Taxonomies', 'widget-options'); ?>"><span class="dashicons dashicons-rest-api"></span> <span class="tabtitle"><?php _e('Taxonomies', 'widget-options'); ?></span></a>
                            </li>
                        <?php } ?>
                        <div class="extended-widget-opts-clearfix"></div>
                    </ul><!--  end tab nav -->
                    <div class="extended-widget-opts-clearfix"></div>

                    <?php if (
                        isset($widget_options['settings']['visibility']) &&
                        isset($widget_options['settings']['visibility']['misc']) &&
                        '1' == $widget_options['settings']['visibility']['misc']
                    ) { ?>
                        <!--  start misc tab content -->
                        <div id="extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-misc" class="extended-widget-opts-visibility-tabcontent extended-widget-opts-inner-tabcontent">
                            <div class="extended-widget-opts-misc extended-widget-opts-inner-lists" style="padding-top: 10px;">
                                <p class="widgetopts-subtitle">Default Pages</p>
                                <?php foreach ($misc as $key => $value) {
                                    if (isset($misc_values[$key]) && $misc_values[$key] == '1') {
                                        $checked = 'checked="checked"';
                                    } else {
                                        $checked = '';
                                    }
                                ?>
                                    <p>
                                        <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][misc][<?php echo $key; ?>]" id="<?php echo $args['id']; ?>-opts-misc-<?php echo $key; ?>" value="1" <?php echo $checked; ?> />
                                        <label for="<?php echo $args['id']; ?>-opts-misc-<?php echo $key; ?>"><?php echo $value; ?></label>
                                    </p>
                                <?php } ?>

                                <h4 class="widgetopts-subtitle" id="extended-widget-opts-pages"><?php _e('Pages', 'widget-options'); ?> +/-<br>
                                    <small>Type atleast 3 characters to initiate the search</small>
                                </h4>
                                <div class="extended-widget-opts-pages extended-widget-opts-parent-option">

                                    <div style="margin-bottom: 10px;">
                                        <button type="button" class="widgetopts-search-option-btn" style="width: 75px; background-color: #3D434A; color: #fff; border-radius: 10px 0px 0 10px; border: 1.5px solid #3D434A;">Search</button>
                                        <button type="button" class="widgetopts-dropdown-option-btn" style="margin-left: -5px;width: 75px;border-radius: 0 10px 10px 0;color: #777A80;background-color: #fff;border: 1.5px solid #777A80;">Checkbox</button>
                                    </div>

                                    <select class="widefat extended-widget-opts-select2-dropdown extended-widget-opts-select2-page-dropdown" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][pages][]" data-namespace="<?php echo $args['namespace']; ?>" multiple="multiple">
                                        <?php
                                        $pargs = array(
                                            'hierarchical' => true,
                                            'child_of' => 0, // Display all pages regardless of parent
                                            'parent' => -1, // Display all pages regardless of parent
                                            'sort_order' => 'ASC',
                                            'sort_column' => 'menu_order, post_title'
                                        );

                                        $pageLoop = get_pages($pargs);

                                        if ($pageLoop) {
                                            foreach ($pageLoop as $objPage) {
                                                $depth = count(get_ancestors($objPage->ID, 'page'));
                                                // Determine indentation for hierarchical display
                                                $indent = str_repeat('-', $depth);

                                                // Check if the page is selected based on form submission
                                                $_selected_page = !empty($pages_values) && in_array($objPage->ID, $pages_values) ? 'selected' : '';

                                                echo '<option value="' . $objPage->ID . '" ' . $_selected_page . '>' . $indent . esc_html($objPage->post_title) . '</option>';
                                            }
                                        }
                                        ?>
                                    </select>


                                    <?php
                                    // $page_class = new WidgetOpts_Pages_Checkboxes();
                                    // $page_checkboxes = $page_class->walk( $widgetopts_pages, 0, $args );
                                    // if ( $page_checkboxes ) {
                                    //     echo '<div class="widgetopts-ul-pages">' . $page_checkboxes . '</div>';
                                    // }
                                    ?>
                                </div>
                            </div>
                        </div><!--  end misc tab content -->
                    <?php } ?>

                    <?php if (
                        isset($widget_options['settings']['visibility']) &&
                        isset($widget_options['settings']['visibility']['post_type']) &&
                        '1' == $widget_options['settings']['visibility']['post_type']
                    ) { ?>
                        <!--  start types tab content -->
                        <div id="extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-types" class="extended-widget-opts-visibility-tabcontent extended-widget-opts-inner-tabcontent extended-widget-opts-tabcontent-pages">
                            <div class="extended-widget-opts-inner-lists" style="height: 230px;padding: 5px;overflow:auto;">


                                <h4 class="widgetopts-subtitle" id="extended-widget-opts-types"><?php _e('Custom Post Types', 'widget-options'); ?> +/-<br></h4>
                                <div class="extended-widget-opts-types">
                                    <?php foreach ($types as $ptype => $type) {
                                        // if ( ! $type->has_archive ) {
                                        //     // don't give the option if there is no archive page
                                        //     continue;
                                        // }

                                        if (isset($types_values[$ptype]) && $types_values[$ptype] == '1') {
                                            $checked = 'checked="checked"';
                                        } else {
                                            $checked = '';
                                        }
                                    ?>
                                        <p>
                                            <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][types][<?php echo $ptype; ?>]" id="<?php echo $args['id']; ?>-opts-types-<?php echo $ptype; ?>" value="1" <?php echo $checked; ?> />
                                            <label for="<?php echo $args['id']; ?>-opts-types-<?php echo $ptype; ?>"><?php echo stripslashes($type->labels->name); ?></label>
                                        </p>
                                    <?php } ?>
                                </div>
                            </div>
                        </div><!--  end types tab content -->
                    <?php } ?>

                    <?php if (
                        isset($widget_options['settings']['visibility']) &&
                        isset($widget_options['settings']['visibility']['taxonomies']) &&
                        '1' == $widget_options['settings']['visibility']['taxonomies']
                    ) {
                        $options_page = 1;
                        if (
                            isset($args['params']) &&
                            isset($args['params']['visibility']) &&
                            isset($args['params']['visibility']['tax_terms_page']) &&
                            isset($args['params']['visibility']['tax_terms_page']['category'])
                        ) {
                            $options_page = $args['params']['visibility']['tax_terms_page']['category'];
                        }
                    ?>
                        <!--  start tax tab content -->
                        <div id="extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-tax" class="extended-widget-opts-visibility-tabcontent extended-widget-opts-inner-tabcontent extended-widget-opts-tabcontent-taxonomies">
                            <div class="extended-widget-opts-inner-lists" style="height: 230px;padding: 5px;overflow:auto;">
                                <h4 id="extended-widget-opts-categories"><?php _e('Categories', 'widget-options'); ?> +/-<br>
                                    <small>Type atleast 3 characters to initiate the search for Category term</small>
                                </h4>
                                <div class="extended-widget-opts-categories extended-widget-opts-parent-option">
                                    <div style="margin-bottom: 10px;">
                                        <button type="button" class="widgetopts-search-option-btn" style="width: 75px; background-color: #3D434A; color: #fff; border-radius: 10px 0px 0 10px; border: 1.5px solid #3D434A;">Search</button>
                                        <button type="button" class="widgetopts-dropdown-option-btn" style="margin-left: -5px;width: 75px;border-radius: 0 10px 10px 0;color: #777A80;background-color: #fff;border: 1.5px solid #777A80;">Checkbox</button>
                                    </div>
                                    <select class="widefat extended-widget-opts-select2-dropdown extended-widget-opts-select2-taxonomy-dropdown" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][tax_terms][category][]" data-taxonomy="category" data-namespace="<?php echo $args['namespace']; ?>" multiple="multiple">
                                        <?php
                                        $taxLoop  = get_terms(['taxonomy' => 'category', 'hide_empty' => false]); //get_terms(['taxonomy' => 'category', 'include' => $term_values, 'hide_empty' => false]);
                                        foreach ($taxLoop as $objTax) {
                                            $term_selected = !empty($term_values) && in_array($objTax->term_id, $term_values) ? 'selected' : '';
                                            echo '<option value="' . $objTax->term_id . '" ' . $term_selected . '>' . $objTax->name . '</option>';
                                        }
                                        ?>
                                    </select>

                                    <p style="margin-top: 10px;">
                                        <strong><?php _e('Select Pages', 'widget-options'); ?></strong><br>
                                        <small><?php _e('Select where to show/hide widget.', 'widget-options'); ?></small><br>
                                        <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][tax_terms_page][category]">
                                            <option value="1" <?php if ($options_page == 1) {
                                                                    echo 'selected="selected"';
                                                                } ?>><?php _e('Archive and Single posts', 'widget-options'); ?></option>
                                            <option value="2" <?php if ($options_page == 2) {
                                                                    echo 'selected="selected"';
                                                                } ?>><?php _e('Archive only', 'widget-options'); ?></option>
                                            <option value="3" <?php if ($options_page == 3) {
                                                                    echo 'selected="selected"';
                                                                } ?>><?php _e('Single posts only', 'widget-options'); ?></option>
                                        </select>
                                    </p>
                                </div>

                                <h4 id="extended-widget-opts-taxonomies"><?php _e('Taxonomies', 'widget-options'); ?> +/-</h4>
                                <div class="extended-widget-opts-taxonomies">
                                    <?php foreach ($taxonomies as $taxonomy) {
                                        if (isset($tax_values[$taxonomy->name]) && $tax_values[$taxonomy->name] == '1') {
                                            $checked = 'checked="checked"';
                                        } else {
                                            $checked = '';
                                        }
                                    ?>
                                        <p>
                                            <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][taxonomies][<?php echo $taxonomy->name; ?>]" id="<?php echo $args['id']; ?>-opts-taxonomies-<?php echo $taxonomy->name; ?>" value="1" <?php echo $checked; ?> />
                                            <label for="<?php echo $args['id']; ?>-opts-taxonomies-<?php echo $taxonomy->name; ?>"><?php echo $taxonomy->label; ?></label> <?php if (isset($taxonomy->object_type) && isset($taxonomy->object_type[0])) {
                                                                                                                                                                                echo ' <small>- ' . $taxonomy->object_type[0] . '</small>';
                                                                                                                                                                            } ?>
                                        </p>
                                    <?php } ?>
                                </div>
                            </div>
                            <?php $upgrade_link = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE)); ?>
                            <div class="extended-widget-opts-feature-warning">
                                <small><?php _e('<em>Upgrade to <a href="' . $upgrade_link . '" target="_blank">Pro Version</a> to manage visibility for Custom Taxonomies.</em>', 'widget-options'); ?></small>
                            </div>
                        </div><!--  end tax tab content -->
                    <?php } ?>
                </div><!--  end .extended-widget-opts-visibility-tabs -->
            </div><!-- End WordPress Pages tab -->
        </div><!--  end main tab -->

    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_visibility');

// Page Options
function widgetopts_ajax_page_search()
{
    global $wp_version;

    $response = [
        'results' => [],
        'pagination' => ['more' => false]
    ];

    if (!empty($_POST['term'])) {
        $args = array(
            'post_type'     => 'page',
            'post_status'   => 'publish',
            's' => $_POST['term'],
        );

        $is_6_3_and_above = version_compare($wp_version, '6.3', '>=');
        if ($is_6_3_and_above) {
            $args['cache_results'] = apply_filters('cache_widgetopts_ajax_page_search', true);
        }

        $query = new WP_Query($args);
        while ($query->have_posts()) {
            $query->the_post();
            $response['results'][] = [
                'id' => get_the_ID(),
                'text' => get_the_title()
            ];
        }
    }

    echo json_encode($response);
    die();
}
add_action('wp_ajax_widgetopts_ajax_page_search',  'widgetopts_ajax_page_search');

// Taxonomy Options
function widgetopts_ajax_taxonomy_search()
{
    $response = [
        'results' => [],
        'pagination' => ['more' => false]
    ];

    if (!empty($_POST['term']) && $_POST['taxonomy']) {
        $args = array(
            'taxonomy'      => array($_POST['taxonomy']),
            'fields'        => 'all',
            'name__like'    => $_POST['term'],
            'hide_empty' => false
        );

        $terms = get_terms($args);
        foreach ($terms as $term) {
            $response['results'][] = [
                'id' => $term->term_id,
                'text' => $term->name
            ];
        }
    }

    echo json_encode($response);
    die();
}
add_action('wp_ajax_widgetopts_ajax_taxonomy_search',  'widgetopts_ajax_taxonomy_search'); ?>