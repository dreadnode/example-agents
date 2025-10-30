<?php

/**
 * Settings Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Settings Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
if (!function_exists('widgetopts_tab_settings')) :
    function widgetopts_tab_settings($args)
    { ?>
        <li class="extended-widget-opts-tab-class">
            <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-class" title="<?php _e('Logic & ACF', 'widget-options'); ?>"><span class="dashicons dashicons-code-standards"></span> <span class="tabtitle"><?php _e('Logic & ACF', 'widget-options'); ?></span></a>
        </li>
    <?php
    }
    add_action('extended_widget_opts_tabs', 'widgetopts_tab_settings', 8);
endif;

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
if (!function_exists('widgetopts_tabcontent_settings')) :
    function widgetopts_tabcontent_settings($args)
    {
        global $widget_options;

        $id         = '';
        $classes    = '';
        $logic      = '';
        $selected   = 0;
        $check      = '';
        if (isset($args['params']) && isset($args['params']['class'])) {
            if (isset($args['params']['class']['id'])) {
                $id = $args['params']['class']['id'];
            }
            if (isset($args['params']['class']['classes'])) {
                $classes = $args['params']['class']['classes'];
            }
            if (isset($args['params']['class']['selected'])) {
                $selected = $args['params']['class']['selected'];
            }
            if (isset($args['params']['class']['logic'])) {
                $logic = $args['params']['class']['logic'];
            }
            if (isset($args['params']['class']['title']) && $args['params']['class']['title'] == '1') {
                $check = 'checked="checked"';
            }
        }

        if (isset($args['params']) && isset($args['params']['visibility'])) {
            if (isset($args['params']['visibility']['acf'])) {
                $acf_values = $args['params']['visibility']['acf'];
            }
        }

        $predefined = array();
        if (isset($widget_options['settings']['classes']) && isset($widget_options['settings']['classes']['classlists']) && !empty($widget_options['settings']['classes']['classlists'])) {
            $predefined = $widget_options['settings']['classes']['classlists'];
        }
    ?>
        <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-class" class="extended-widget-opts-tabcontent extended-widget-opts-inside-tabcontent extended-widget-opts-tabcontent-class">

            <div class="extended-widget-opts-settings-tabs extended-widget-opts-inside-tabs">
                <input type="hidden" id="extended-widget-opts-settings-selectedtab" value="<?php echo $selected; ?>" name="<?php echo $args['namespace']; ?>[extended_widget_opts][class][selected]" />
                <!--  start tab nav -->
                <ul style="margin-top: 10px;" class="extended-widget-opts-settings-tabnav-ul">

                    <?php if ('activate' == $widget_options['logic'] && current_user_can('administrator')) { ?>
                        <li class="extended-widget-opts-settings-tab-logic">
                            <a href="#extended-widget-opts-settings-tab-<?php echo $args['id']; ?>-logic" title="<?php _e('Display Logic', 'widget-options'); ?>"><?php _e('Logic', 'widget-options'); ?></a>
                        </li>
                    <?php } ?>

                    <?php if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) { ?>
                        <li class="extended-widget-opts-visibility-tab-acf">
                            <a href="#extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-acf" title="<?php _e('ACF', 'widget-options'); ?>"><?php _e('ACF', 'widget-options'); ?></a>
                        </li>
                    <?php } ?>
                    <div class="extended-widget-opts-clearfix"></div>
                </ul><!--  end tab nav -->
                <div class="extended-widget-opts-clearfix"></div>

                <?php if ('activate' == $widget_options['logic'] && current_user_can('administrator')) { ?>
                    <!--  start logic tab content -->
                    <div id="extended-widget-opts-settings-tab-<?php echo $args['id']; ?>-logic" class="extended-widget-opts-settings-tabcontent extended-widget-opts-inner-tabcontent">
                        <div class="widget-opts-logic">
                            <p><small><?php _e('The text field lets you use <a href="http://codex.wordpress.org/Conditional_Tags" target="_blank">WP Conditional Tags</a>, or any general PHP code.', 'widget-options'); ?></small></p>
                            <textarea class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][class][logic]"><?php echo stripslashes($logic); ?></textarea>

                            <?php if (
                                !isset($widget_options['settings']['logic']) ||
                                (isset($widget_options['settings']['logic']) && !isset($widget_options['settings']['logic']['notice']))
                            ) { ?>
                                <p><a href="#" class="widget-opts-toggler-note"><?php _e('Click to Toggle Note', 'widget-options'); ?></a></p>
                                <p class="widget-opts-toggle-note"><small><?php _e('PLEASE NOTE that the display logic you introduce is EVAL\'d directly. Anyone who has access to edit widget appearance will have the right to add any code, including malicious and possibly destructive functions. There is an optional filter <em>"widget_options_logic_override"</em> which you can use to bypass the EVAL with your own code if needed.', 'widget-options'); ?></small></p>
                            <?php } ?>
                        </div>
                    </div><!--  end logiv tab content -->
                <?php } ?>

                <!-- Start ACF tab -->
                <?php if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) : ?>

                    <?php if (isset($widget_options['acf']) && 'activate' == $widget_options['acf']) : ?>
                        <div id="extended-widget-opts-visibility-tab-<?php echo $args['id']; ?>-acf" class="extended-widget-opts-visibility-tabcontent extended-widget-opts-inside-tabcontent extended-widget-opts-inner-tabcontent">
                            <p class="widgetopts-subtitle" style="display: none !important;"><?php _e('ACF', 'widget-options'); ?></p>
                            <?php
                            $fields = array();

                            if (function_exists('acf_get_field_groups')) {
                                $groups = acf_get_field_groups();
                                if (is_array($groups)) {
                                    foreach ($groups as $group) {
                                        $fields[$group['ID']] = array('title' => $group['title'], 'fields' => acf_get_fields($group));
                                    }
                                }
                            } else {
                                $groups = apply_filters('acf/get_field_groups', array());
                                if (is_array($groups)) {
                                    foreach ($groups as $group) {
                                        $fields[$group['id']] = array('title' => $group['title'], 'fields' => apply_filters('acf/field_group/get_fields', array(), $group['id']));
                                    }
                                }
                            }
                            ?>
                            <p><strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
                                <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][acf][visibility]">
                                    <option value="hide" <?php echo (isset($acf_values['visibility']) && $acf_values['visibility'] == 'hide') ? 'selected="selected"' : '' ?>><?php _e('Hide when Condition\'s met', 'widget-options'); ?></option>
                                    <option value="show" <?php echo (isset($acf_values['visibility']) && $acf_values['visibility'] == 'show') ? 'selected="selected"' : '' ?>><?php _e('Show when Condition\'s met', 'widget-options'); ?></option>
                                </select>
                            </p>

                            <p><strong><?php _e('Choose ACF Field', 'widget-options'); ?></strong>
                                <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][acf][field]">
                                    <option value=""><?php _e('Select Field', 'widget-options'); ?></option>
                                    <?php
                                    if (!empty($fields)) {
                                        foreach ($fields as $k => $field) { ?>
                                            <optgroup label="<?php echo $field['title']; ?>">
                                                <?php foreach ($field['fields'] as $key => $f) { ?>
                                                    <option value="<?php echo $f['key']; ?>" <?php echo (isset($acf_values['field']) && $acf_values['field'] == $f['key']) ? 'selected="selected"' : '' ?>><?php echo $f['label']; ?></option>
                                                <?php } ?>
                                            </optgroup>
                                    <?php }
                                    } ?>
                                </select>
                            </p>
                            <p><strong><?php _e('Condition', 'widget-options'); ?></strong>
                                <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][acf][condition]">
                                    <option value=""><?php _e('Select Condition', 'widget-options'); ?></option>
                                    <optgroup label="<?php _e('Conditional', 'widget-options'); ?>">
                                        <option value="equal" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'equal') ? 'selected="selected"' : '' ?>><?php _e('Is Equal to', 'widget-options'); ?></option>
                                        <option value="not_equal" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'not_equal') ? 'selected="selected"' : '' ?>><?php _e('Is Not Equal to', 'widget-options'); ?></option>
                                        <option value="contains" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'contains') ? 'selected="selected"' : '' ?>><?php _e('Contains', 'widget-options'); ?></option>
                                        <option value="not_contains" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'not_contains') ? 'selected="selected"' : '' ?>><?php _e('Does Not Contain', 'widget-options'); ?></option>
                                    </optgroup>
                                    <optgroup label="<?php _e('Value Based', 'widget-options'); ?>">
                                        <option value="empty" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'empty') ? 'selected="selected"' : '' ?>><?php _e('Is Empty', 'widget-options'); ?></option>
                                        <option value="not_empty" <?php echo (isset($acf_values['condition']) && $acf_values['condition'] == 'not_empty') ? 'selected="selected"' : '' ?>><?php _e('Is Not Empty', 'widget-options'); ?></option>
                                    </optgroup>
                                </select>
                            </p>
                            <p><strong><?php _e('Conditional Value', 'widget-options'); ?></strong>
                                <textarea name="<?php echo $args['namespace']; ?>[extended_widget_opts][visibility][acf][value]" id="<?php echo $args['id']; ?>-opts-acf-value" class="widefat widgetopts-acf-conditional"><?php echo (isset($acf_values['value'])) ? $acf_values['value'] : '' ?></textarea>
                            </p>
                        </div>
                    <?php endif; ?>
                <?php endif; ?>
                <!-- End ACF tab -->

            </div><!-- end .extended-widget-opts-settings-tabs -->


        </div>
<?php
    }
    add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_settings');
endif; ?>