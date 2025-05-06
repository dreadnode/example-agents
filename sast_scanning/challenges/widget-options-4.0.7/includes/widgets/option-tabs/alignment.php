<?php

/**
 * Alignment Widget Options
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
function widgetopts_tab_alignment($args)
{ ?>
    <li class="extended-widget-opts-tab-alignment">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-alignment" title="<?php _e('Alignment', 'widget-options'); ?>"><span class="dashicons dashicons-editor-aligncenter"></span> <span class="tabtitle"><?php _e('Alignment', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_alignment', 2);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
function widgetopts_tabcontent_alignment($args)
{
    $desktop = '';
    if (isset($args['params']) && isset($args['params']['alignment'])) {
        if (isset($args['params']['alignment']['desktop'])) {
            $desktop = $args['params']['alignment']['desktop'];
        }
    }
    $upgrade_link = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE));
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-alignment" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-alignment">
        <table class="form-table">
            <tbody>
                <tr valign="top">
                    <td scope="row"><strong><?php _e('Devices', 'widget-options'); ?></strong></td>
                    <td><strong><?php _e('Alignment', 'widget-options'); ?></strong></td>
                </tr>
                <tr valign="top">
                    <td scope="row"><span class="dashicons dashicons-desktop"></span> <?php _e('All Devices', 'widget-options'); ?></td>
                    <td>
                        <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][alignment][desktop]">
                            <option value="default"><?php _e('Default', 'widget-options'); ?></option>
                            <option value="center" <?php if ($desktop == 'center') {
                                                        echo 'selected="selected"';
                                                    } ?>><?php _e('Center', 'widget-options'); ?></option>
                            <option value="left" <?php if ($desktop == 'left') {
                                                        echo 'selected="selected"';
                                                    } ?>><?php _e('Left', 'widget-options'); ?></option>
                            <option value="right" <?php if ($desktop == 'right') {
                                                        echo 'selected="selected"';
                                                    } ?>><?php _e('Right', 'widget-options'); ?></option>
                            <option value="justify" <?php if ($desktop == 'justify') {
                                                        echo 'selected="selected"';
                                                    } ?>><?php _e('Justify', 'widget-options'); ?></option>
                        </select>
                    </td>
                </tr>
                <tr valign="top" class="widgetopts-topro">
                    <td colspan="2">
                        <div class="extended-widget-opts-feature-warning">
                            <small><?php _e('<em>Upgrade to <a href="' . $upgrade_link . '" target="_blank">Pro Version</a> for Multiple Devices Alignment and Additional Widget Options.</em>', 'widget-options'); ?></small>
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_alignment'); ?>