<?php

/**
 * Styling Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Styling Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_styling($args)
{ ?>
    <li class="extended-widget-opts-tab-styling">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-styling" title="<?php _e('Styling', 'widget-options'); ?>"><span class="dashicons dashicons-art"></span> <span class="tabtitle"><?php _e('Styling', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_styling', 6);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
function widgetopts_tabcontent_styling($args)
{
    global $widget_options;

    $selected               = 0;
    $bg_image               = '';
    $background             = '';
    $background_hover       = '';
    $heading                = '';
    $text                   = '';
    $links                  = '';
    $links_hover            = '';
    $border_color           = '';
    $border_width           = '';
    $border_type            = '';

    $background_input       = '';
    $text_input             = '';
    $border_color_input     = '';
    $border_width_input     = '';
    $border_type_input      = '';

    $background_submit      = '';
    $background_submit_hover  = '';
    $text_submit            = '';
    $border_color_submit    = '';
    $border_width_submit    = '';
    $border_type_submit     = '';

    $list_border_color      = '';
    $table_border_color     = '';
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-styling" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-styling">
        <div class="extended-widget-opts-demo-feature">
            <div class="extended-widget-opts-demo-warning">
                <p class="widgetopts-unlock-features">
                    <span class="dashicons dashicons-lock"></span><br>
                    Unlock all Features<br>
                    <a href="https://widget-options.com/?utm_source=wordpressadmin&amp;utm_medium=widgettabs&amp;utm_campaign=widgetoptsprotab" class="button-primary" target="_blank">Learn More</a>
                </p>
            </div>

            <div class="extended-widget-opts-styling-tabs extended-widget-opts-inside-tabs">

                <ul class="extended-widget-opts-styling-tabnav-ul">
                    <li class="extended-widget-opts-styling-tab-styling">
                        <a href="#extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-widget"><?php _e('Widget', 'widget-options'); ?></a>
                    </li>
                    <li class="extended-widget-opts-styling-tab-form">
                        <a href="#extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-form"><?php _e('Forms', 'widget-options'); ?></a>
                    </li>
                    <li class="extended-widget-opts-styling-tab-form">
                        <a href="#extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-others"><?php _e('Others', 'widget-options'); ?></a>
                    </li>
                    <div class="extended-widget-opts-clearfix"></div>
                </ul>

                <div id="extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-widget" class="extended-widget-opts-styling-tabcontent extended-widget-opts-inner-tabcontent">
                    <p class="widgetopts-subtitle"><?php _e('Background Image', 'widget-options'); ?></p>

                    <table class="form-table">
                        <tbody>
                            <tr valign="top">
                                <td colspan="2"><input type="text" class="widefat extended_widget_opts-bg-image" name="<?php echo $args['namespace']; ?>[extended_widget_opts][styling][bg_image]" value="<?php echo $bg_image; ?>" placeholder="<?php _e('Image Url', 'widget-options'); ?>" />
                                </td>
                            </tr>
                            <tr valign="top">
                                <td colspan="2" class="alright">
                                    <input type="button" class="button-primary extended_widget_opts-bg_uploader" value="<?php _e('Upload', 'widget-options'); ?>">
                                    <input type="button" class="button-secondary extended_widget_opts-remove_image" value="<?php _e('Remove', 'widget-options'); ?>">
                                </td>
                            </tr>
                        </tbody>
                    </table><br />

                    <p class="widgetopts-subtitle"><?php _e('Widget Styling Options', 'widget-options'); ?></p>

                    <table class="form-table">
                        <tbody>
                            <tr valign="top">
                                <td scope="row"><?php _e('Background Color', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Hover Background Color', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Headings', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Text', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Links', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Links Hover', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Border Color', 'widget-options'); ?></td>
                                <td><input type="text" class="widget-opts-color" readonly /></td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Border Style', 'widget-options'); ?></td>
                                <td>
                                    <select class="widefat" readonly>
                                        <option value=""><?php _e('Default', 'widget-options'); ?></option>
                                    </select>
                                </td>
                            </tr>
                            <tr valign="top">
                                <td scope="row"><?php _e('Border Width', 'widget-options'); ?></td>
                                <td><input type="text" size="5" class="inputsize5" readonly />px</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div id="extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-form" class="extended-widget-opts-styling-tabcontent extended-widget-opts-inner-tabcontent"></div>

                <div id="extended-widget-opts-styling-tab-<?php echo $args['id']; ?>-others" class="extended-widget-opts-styling-tabcontent extended-widget-opts-inner-tabcontent"></div>

                <div class="extended-widget-opts-clearfix"></div>
            </div><!--  end .extended-widget-opts-tabs -->
        </div>

    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_styling'); ?>