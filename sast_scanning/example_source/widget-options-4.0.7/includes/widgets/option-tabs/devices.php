<?php

/**
 * Devices Visibility Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Alignment Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_devices($args)
{ ?>
    <li class="extended-widget-opts-tab-devices">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-devices" title="<?php _e('Devices', 'widget-options'); ?>"><span class="dashicons dashicons-smartphone"></span> <span class="tabtitle"><?php _e('Devices', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_devices', 4);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for devices visibility options
 */
function widgetopts_tabcontent_devices($args)
{
    $desktop        = '';
    $tablet         = '';
    $mobile         = '';
    $options_role   = '';
    if (isset($args['params']) && isset($args['params']['devices'])) {
        if (isset($args['params']['devices']['options'])) {
            $options_role = $args['params']['devices']['options'];
        }
        if (isset($args['params']['devices']['desktop'])) {
            $desktop = $args['params']['devices']['desktop'];
        }
        if (isset($args['params']['devices']['tablet'])) {
            $tablet = $args['params']['devices']['tablet'];
        }
        if (isset($args['params']['devices']['mobile'])) {
            $mobile = $args['params']['devices']['mobile'];
        }
    }
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-devices" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-devices">
        <p>
            <strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
            <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][devices][options]">
                <option value="hide" <?php if ($options_role == 'hide') {
                                            echo 'selected="selected"';
                                        } ?>><?php _e('Hide on checked devices', 'widget-options'); ?></option>
                <option value="show" <?php if ($options_role == 'show') {
                                            echo 'selected="selected"';
                                        } ?>><?php _e('Show on checked devices', 'widget-options'); ?></option>
            </select>
        </p>
        <table class="form-table">
            <tbody>
                <tr valign="top">
                    <td scope="row"><strong><?php _e('Devices', 'widget-options'); ?></strong></td>
                    <td>&nbsp;</td>
                </tr>
                <tr valign="top">
                    <td scope="row"><span class="dashicons dashicons-desktop"></span> <label for="extended_widget_opts-<?php echo $args['id']; ?>-devices-desktop"><?php _e('Desktop', 'widget-options'); ?></label></td>
                    <td>
                        <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][devices][desktop]" value="1" id="extended_widget_opts-<?php echo $args['id']; ?>-devices-desktop" <?php if (!empty($desktop)) {
                                                                                                                                                                                                                    echo 'checked="checked"';
                                                                                                                                                                                                                } ?> />
                    </td>
                </tr>
                <tr valign="top">
                    <td scope="row"><span class="dashicons dashicons-tablet"></span> <label for="extended_widget_opts-<?php echo $args['id']; ?>-devices-table"><?php _e('Tablet', 'widget-options'); ?></label></td>
                    <td>
                        <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][devices][tablet]" value="1" id="extended_widget_opts-<?php echo $args['id']; ?>-devices-table" <?php if (!empty($tablet)) {
                                                                                                                                                                                                                echo 'checked="checked"';
                                                                                                                                                                                                            } ?> />
                    </td>
                </tr>
                <tr valign="top">
                    <td scope="row"><span class="dashicons dashicons-smartphone"></span> <label for="extended_widget_opts-<?php echo $args['id']; ?>-devices-mobile"><?php _e('Mobile', 'widget-options'); ?></label></td>
                    <td>
                        <input type="checkbox" name="<?php echo $args['namespace']; ?>[extended_widget_opts][devices][mobile]" value="1" id="extended_widget_opts-<?php echo $args['id']; ?>-devices-mobile" <?php if (!empty($mobile)) {
                                                                                                                                                                                                                echo 'checked="checked"';
                                                                                                                                                                                                            } ?> />
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_devices'); ?>