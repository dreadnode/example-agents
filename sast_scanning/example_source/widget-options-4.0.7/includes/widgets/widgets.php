<?php

/**
 * Add Widget Options
 *
 * Process Managing of Widget Options.
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       2.0
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;



/**
 * Output widget search filter textfield before widget lists
 *
 * @since  3.3
 */
if (!function_exists('widgetopts_add_search_input')) :
    function widgetopts_add_search_input()
    {
        global $widget_options;
        if (isset($widget_options['search']) && 'activate' == $widget_options['search']) : ?>
            <div id="widgetopts-widgets-filter">
                <label class="screen-reader-text" for="widgetopts-widgets-search"><?php _e('Search Widgets', 'widget-options'); ?></label>
                <input type="text" id="widgetopts-widgets-search" class="widgetopts-widgets-search" placeholder="<?php esc_attr_e('Search widgets&hellip;', 'widget-options') ?>" />
                <div class="widgetopts-search-icon" aria-hidden="true"></div>
                <button type="button" class="widgetopts-clear-results"><span class="screen-reader-text"><?php _e('Clear Results', 'widget-options'); ?></span></button>
                <p class="screen-reader-text" id="widgetopts-search-desc"><?php _e('The search results will be updated as you type.', 'widget-options'); ?></p>
            </div>
    <?php
        endif;
    }
    add_action('widgets_admin_page', 'widgetopts_add_search_input');
endif;

/**
 * Add Options on in_widget_form action
 *
 * @since 2.0
 * @return void
 */

function widgetopts_in_widget_form($widget, $return, $instance)
{
    global $widget_options, $wp_registered_widget_controls;
    $width          = (isset($wp_registered_widget_controls[$widget->id]['width'])) ? (int) $wp_registered_widget_controls[$widget->id]['width']  : 250;
    $opts           = (isset($instance['extended_widget_opts-' . $widget->id]))    ? $instance['extended_widget_opts-' . $widget->id]             : array();
    $is_siteorigin  = (isset($widget_options['siteorigin'])) ? $widget_options['siteorigin'] : '';

    /* if $opts is empty, try to get data from blocks */
    if (!wp_use_widgets_block_editor()) {
        if (empty($instance['extended_widget_opts-' . $widget->id])) {
            if (isset($instance['content']) && !empty($instance['content'])) {
                $block = parse_blocks($instance['content']);
                if (!empty($block[0]) && !empty($block[0]['attrs'])) {
                    if (!empty($block[0]['attrs']['extended_widget_opts'])) {
                        $opts = $block[0]['attrs']['extended_widget_opts'];
                    }
                }
            }
        }
    }

    /** change widget names for SO Pagebuilder support **/
    if (isset($widget->id) && 'temp' == $widget->id) {
        $namespace  = 'widgets[' . $widget->number . ']';
        $optsname   = 'widgets[' . $widget->number . '][extended_widget_opts_name]';
        $opts       = (isset($instance['extended_widget_opts'])) ? $instance['extended_widget_opts'] : array();
        $widget->id = $widget->number;

        //create siteorigin pagebuilder variable
        echo '<input type="hidden" name="' . $namespace . '[siteorigin]" value="1" />';
    } else {
        $namespace = 'extended_widget_opts-' . $widget->id;
        $optsname   = 'extended_widget_opts_name';
    }

    $args = array(
        'width'     =>  $width,
        'id'        =>  $widget->id,
        'params'    =>  $opts,
        'namespace' =>  $namespace
    );
    $selected = 0;
    if (isset($opts['tabselect'])) {
        $selected = $opts['tabselect'];
    }

    ?>

    <input type="hidden" name="extended_widget_opts_name" value="extended_widget_opts-<?php echo $widget->id; ?>">
    <input type="hidden" name="<?php echo $args['namespace']; ?>[extended_widget_opts][id_base]" value="<?php echo $widget->id; ?>" />
    <div class="extended-widget-opts-form <?php if ($width > 480) {
                                                echo 'extended-widget-opts-form-large';
                                            } else if ($width <= 480) {
                                                echo 'extended-widget-opts-form-small';
                                            } ?>">
        <div class="extended-widget-opts-tabs">
            <ul class="extended-widget-opts-tabnav-ul">
                <?php do_action('extended_widget_opts_tabs', $args); ?>
                <div class="extended-widget-opts-clearfix"></div>
            </ul>

            <?php do_action('extended_widget_opts_tabcontent', $args); ?>
            <input type="hidden" id="extended-widget-opts-selectedtab" value="<?php echo $selected; ?>" name="extended_widget_opts-<?php echo $args['id']; ?>[extended_widget_opts][tabselect]" />
            <div class="extended-widget-opts-clearfix"></div>
        </div><!--  end .extended-widget-opts-tabs -->
    </div><!-- end .extended-widget-opts-form -->

    <?php if ('activate' == $is_siteorigin) { ?>
        <script type="text/javascript">
            jQuery(document).ready(function($) {
                if ($('.so-content .extended-widget-opts-tabs').length > 0) {
                    $('.extended-widget-opts-tabs').tabs({
                        active: 1
                    });
                    $('.extended-widget-opts-visibility-tabs').tabs({
                        active: 0
                    });
                    $('.extended-widget-opts-settings-tabs').tabs({
                        active: 0
                    });
                }
            });
        </script>
    <?php } else { ?>
        <style type="text/css">
            .so-content.panel-dialog .extended-widget-opts-form {
                display: none;
            }
        </style>
    <?php } ?>

<?php
}
add_action('in_widget_form', 'widgetopts_in_widget_form', 10, 3);

/*
 * Update Options
 */
function widgetopts_ajax_update_callback($instance, $new_instance, $this_widget)
{
    global $widget_options;

    if (
        isset($_POST['extended_widget_opts_name']) ||
        (!isset($_POST['extended_widget_opts_name']) && isset($new_instance['siteorigin']))
    ) {
        //check if from SO pagebuilder
        if (is_array($new_instance) && isset($new_instance['extended_widget_opts']) && isset($new_instance['siteorigin'])) {
            $name       = 'extended_widget_opts';
            $options    = widgetopts_sanitize_array($new_instance);
        } else {
            $name         = strip_tags($_POST['extended_widget_opts_name']);
            $options     = $_POST[$name];
        }
        if (isset($options['extended_widget_opts'])) {
            // update_option( $name , $options['extended_widget_opts'] );
            if (isset($options['extended_widget_opts']['class']['link']) && !empty($options['extended_widget_opts']['class']['link'])) {
                $options['extended_widget_opts']['class']['link'] = widgetopts_addhttp($options['extended_widget_opts']['class']['link']);
            }
            $instance[$name] = widgetopts_sanitize_array($options['extended_widget_opts']);

            //remove cache
            if (isset($options['extended_widget_opts']['id_base']) && isset($widget_options['cache']) && 'activate' == $widget_options['cache']) {
                $transient_name = 'widgetopts-cache_' . $options['extended_widget_opts']['id_base'];
                delete_transient($transient_name);
            }

            //remove widgetopts attribute from blocks when it is classic editor
            if (!empty($instance['content'])) {
                $block = parse_blocks($instance['content']);
                if (!empty($block[0]) && !empty($block[0]['attrs'])) {
                    if (!empty($block[0]['attrs']['extended_widget_opts'])) {
                        unset($block[0]['attrs']['extended_widget_opts']);
                        $instance['content'] = serialize_blocks($block);
                    }
                }
            }
        }
    }
    return $instance;
}
add_filter('widget_update_callback', 'widgetopts_ajax_update_callback', 10, 3);

add_filter('widget_form_callback', function ($instance, $widget) {
    /* if $opts is empty, try to get data from blocks */
    if (!wp_use_widgets_block_editor()) {
        if (empty($instance['extended_widget_opts-' . $widget->id])) {
            if (isset($instance['content']) && !empty($instance['content'])) {
                $block = parse_blocks($instance['content']);
                if (!empty($block[0]) && !empty($block[0]['attrs'])) {
                    if (!empty($block[0]['attrs']['extended_widget_opts'])) {
                        $instance['extended_widget_opts-' . $widget->id] = $block[0]['attrs']['extended_widget_opts'];
                    }
                }
            }
        }

        //remove widgetopts attribute from blocks when it is classic editor
        if (!empty($instance['content'])) {
            $blocks = parse_blocks($instance['content']);
            if (is_array($blocks) && is_iterable($blocks)) {

                foreach ($blocks as &$block) {
                    if (!empty($block) && !empty($block['attrs'])) {
                        $is_there_a_changes = false;
                        $_block = widgetopts_unset_block_attributes($block);
                        if ($_block !== false) {
                            $block = $_block;
                            $is_there_a_changes = true;
                        }

                        //inner blocks
                        if (isset($block['innerBlocks']) && is_array($block['innerBlocks']) && !empty($block['innerBlocks'])) {

                            foreach ($block['innerBlocks'] as &$block2) {
                                $_block2 = widgetopts_unset_block_attributes($block2);
                                if ($_block2 !== false) {
                                    $block2 = $_block2;

                                    $is_there_a_changes = true;
                                }

                                //2nd level inner blocks
                                if (isset($block2['innerBlocks']) && is_array($block2['innerBlocks']) && !empty($block2['innerBlocks'])) {

                                    foreach ($block2['innerBlocks'] as &$block3) {
                                        $_block3 = widgetopts_unset_block_attributes($block3);
                                        if ($_block3 !== false) {
                                            $block3 = $_block3;
                                            $is_there_a_changes = true;
                                        }
                                    }
                                }
                            }
                        }

                        if ($is_there_a_changes) {
                            $instance['content'] = serialize_blocks($blocks);
                            break;
                        }
                    }
                }
            }
        }
    }

    return $instance;
}, 100, 2);

function widgetopts_unset_block_attributes($block)
{
    $is_there_a_changes = false;
    if (!empty($block) && !empty($block['attrs'])) {
        if (!empty($block['attrs']['extended_widget_opts'])) {
            unset($block['attrs']['extended_widget_opts']);
            $is_there_a_changes = true;
        }

        if (!empty($block['attrs']['extended_widget_opts_state'])) {
            unset($block['attrs']['extended_widget_opts_state']);
            $is_there_a_changes = true;
        }

        if (!empty($block['attrs']['extended_widget_opts_clientid'])) {
            unset($block['attrs']['extended_widget_opts_clientid']);
            $is_there_a_changes = true;
        }
    }

    return $is_there_a_changes ? $block : false;
}
?>